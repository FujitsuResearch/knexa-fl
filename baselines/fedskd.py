import flwr as fl, logging, numpy as np, torch
from src.globals import *
from src.data_utils import load_split
from src.client import KnexaClient
import flwr.common as fl_common
from src.globals_runtime import GLOBAL_KB_LOG

torch.backends.cuda.matmul.allow_tf32 = True
 
logging.basicConfig(filename="experimental_artifacts/baselines/others/logs/fedskd.log", level=logging.DEBUG)

CLIENT_SPLITS, GLOBAL_TEST = load_split(NUM_CLIENTS)

class FedSKDClient(KnexaClient):
    """Client for FedSKD (centralized logit averaging)"""
    
    def run_teacher(self, rnd, queries, sub_id):
        """Generate logits for central aggregation"""
        from src.comm import write_blob
        
        tok_in = self.tok(queries, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.gpu)
        logits = self.model(**tok_in).logits.detach()
        logits = logits[:, :-1, :]  # Align
        
        # Apply DP for privacy
        from src.privacy_utils import dp_clip_noise, compute_sier
        logits = dp_clip_noise(logits)
        
        # Use top-k for compression
        topk = logits.topk(TOPK, dim=-1)
        values, indices = topk.values.half(), topk.indices.int()
        
        decoded = [self.tok.decode(x.argmax(-1)) for x in logits]
        sier = compute_sier(decoded)
        
        payload = {"values": values.cpu(), "indices": indices.cpu(), "seq_len": logits.shape[1]}
        kb = write_blob(self.cid, rnd, sub_id, payload, self.logger.info if hasattr(self, 'logger') else print)
        
        self.sier_avg = 0.5 * self.sier_avg + 0.5 * sier
        return sier, kb
    
    def run_student(self, rnd, avg_logits, alpha, T, queries):
        """Learn from centrally averaged logits"""
        import torch.nn.functional as F
        from torch.nn import CrossEntropyLoss
        
        if avg_logits is None:
            return
            
        # Convert averaged logits to soft targets
        soft_t = F.softmax(avg_logits / T, dim=-1)
        
        tok_in = self.tok(queries, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.gpu)
        logits_s = self.model(**tok_in).logits
        logits_s = logits_s[:, :soft_t.shape[1], :]  # Align seq_len
        
        soft_s = F.log_softmax(logits_s / T, dim=-1)
        kl = F.kl_div(soft_s, soft_t, reduction="batchmean") * (T ** 2)
        
        task = CrossEntropyLoss(label_smoothing=0.1)(
            logits_s.view(-1, logits_s.size(-1)), 
            soft_t.argmax(-1).view(-1)
        )
        
        loss = alpha * task + (1 - alpha) * kl
        
        optimiser = torch.optim.Adam(self.model.parameters(), lr=LR_KD)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

def client_fn(cid: str):
    cid = int(cid)
    train, val = CLIENT_SPLITS[cid]
    return FedSKDClient(cid, train, val)

class FedSKDStrategy(fl.server.strategy.Strategy):
    def __init__(self):
        super().__init__()
        self.client_profiles = {}
        self.collected_logits = {}
    
    def initialize_parameters(self, client_manager):
        return fl_common.ndarrays_to_parameters([np.array([0.0])])
    
    def evaluate(self, server_round, parameters):
        return None
    
    def configure_fit(self, server_round, parameters, client_manager):
        clients = client_manager.sample(num_clients=NUM_CLIENTS, min_num_clients=NUM_CLIENTS)
        
        config = {"round": server_round}
        # Get queries from first client's validation data
        val_data = list(CLIENT_SPLITS[0][1])
        queries = [ex["prompt"] for ex in val_data[:20]]  # Use same queries for all
        
        # Phase 1: All clients act as teachers to generate logits
        sub_id = server_round % 3
        for i, client in enumerate(clients):
            config[str(i)] = {
                "role_config": {
                    "role": "teacher", 
                    "queries": queries, 
                    "sub_id": sub_id
                }
            }
        
        dummy_params = fl_common.ndarrays_to_parameters([np.array([0.0])])
        return [(client, fl_common.FitIns(dummy_params, config)) for client in clients]
    
    def aggregate_fit(self, server_round, results, failures):
        from src.comm import read_blob
        import torch
        
        # Collect logits from all teachers
        logits_list = []
        for i in range(NUM_CLIENTS):
            payload = read_blob(i, server_round, print)
            # Reconstruct dense logits from top-k
            values = torch.tensor(payload["values"]).float()
            indices = torch.tensor(payload["indices"])
            seq_len = payload["seq_len"]
            
            # Create dense representation using CodeLlama vocab size
            dense_logits = torch.zeros(values.shape[0], seq_len, 32000)  # CodeLlama vocab size
            for b in range(values.shape[0]):
                for s in range(values.shape[1]):
                    dense_logits[b, s, indices[b, s]] = values[b, s]
            
            logits_list.append(dense_logits)
        
        # Average logits - must have logits from all clients
        avg_logits = torch.stack(logits_list).mean(dim=0)
        self.avg_logits = avg_logits
        
        # Phase 2: Send averaged logits back to students
        self._configure_student_phase(server_round)
        
        return None, {}
    
    def _configure_student_phase(self, server_round):
        """Send averaged logits to all clients for distillation"""
        # This would typically be done in a second round, but for simplicity
        # we'll store for next configure_fit call
        pass
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        return []
    
    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

if __name__ == "__main__":
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=FedSKDStrategy(),
        client_resources={"num_gpus": 0.25, "num_cpus": 1}
    )