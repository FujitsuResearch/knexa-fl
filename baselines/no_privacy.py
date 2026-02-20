import flwr as fl, logging, numpy as np, torch
from src.globals import *
from src.data_utils import load_split
from src.client import KnexaClient
from src.bandit import LinUCB
import flwr.common as fl_common
from src.globals_runtime import GLOBAL_KB_LOG

torch.backends.cuda.matmul.allow_tf32 = True
 
logging.basicConfig(filename="experimental_artifacts/baselines/others/logs/no_privacy.log", level=logging.DEBUG)

CLIENT_SPLITS, GLOBAL_TEST = load_split(NUM_CLIENTS)
bandit = LinUCB(d=32)  # 32-dim because we concatenate two 16-dim profiles

class NoPrivacyClient(KnexaClient):
    """Client with privacy protections disabled (no DP, no SIER, full vocab)"""
    
    def run_teacher(self, rnd, queries, sub_id):
        from src.comm import write_blob
        
        tok_in = self.tok(queries, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.gpu)
        logits = self.model(**tok_in).logits.detach()
        logits = logits[:, :-1, :]  # Align
        
        # No DP clipping or noise
        # Use full vocabulary instead of top-k
        values = logits  # Full logits
        indices = torch.arange(logits.shape[-1]).unsqueeze(0).unsqueeze(0).expand(logits.shape[0], logits.shape[1], -1).to(logits.device)
        
        # Skip SIER computation
        sier = 0.0
        
        payload = {"values": values.cpu(), "indices": indices.cpu(), "seq_len": logits.shape[1]}
        kb = write_blob(self.cid, rnd, sub_id, payload, self.logger.info if hasattr(self, 'logger') else print)
        
        self.sier_avg = 0.5 * self.sier_avg + 0.5 * sier
        return sier, kb

def client_fn(cid: str):
    cid = int(cid)
    train, val = CLIENT_SPLITS[cid]
    return NoPrivacyClient(cid, train, val)

class NoPrivacyStrategy(fl.server.strategy.Strategy):
    def __init__(self):
        super().__init__()
        self.client_profiles = {}
    
    def initialize_parameters(self, client_manager):
        return fl_common.ndarrays_to_parameters([np.array([0.0])])
    
    def evaluate(self, server_round, parameters):
        return None
    
    def configure_fit(self, server_round, parameters, client_manager):
        clients = client_manager.sample(num_clients=NUM_CLIENTS, min_num_clients=NUM_CLIENTS)
        
        # Use profiles from previous round or initialize with zeros
        profiles = []
        for i, client in enumerate(clients):
            cid = int(str(client).split('_')[-1]) if '_' in str(client) else i
            profile = self.client_profiles.get(cid, np.zeros(16))
            profiles.append(profile)
        
        pairs = bandit.choose_pairs(profiles, NUM_CLIENTS//2, server_round)
        config = {"round": server_round}
        
        # Set up KD exchanges
        for i, j, alpha, T in pairs:
            val_data = list(CLIENT_SPLITS[i][1])
            queries = [ex["prompt"] for ex in val_data[:20]]
            sub_id = server_round % 3
            config[str(i)] = {"role_config": {"role": "student", "teacher_cid": j, "alpha": alpha, "T": T, "queries": queries}}
            config[str(j)] = {"role_config": {"role": "teacher", "queries": queries, "sub_id": sub_id}}
        
        dummy_params = fl_common.ndarrays_to_parameters([np.array([0.0])])
        return [(client, fl_common.FitIns(dummy_params, config)) for client in clients]
    
    def aggregate_fit(self, server_round, results, failures):
        for _, fit_res in results:
            metrics = fit_res.metrics
            if "profile" in metrics:
                # Extract client ID and store profile
                cid = fit_res.metrics.get("client_id", 0)  # Fallback
                self.client_profiles[cid] = metrics["profile"]
                
                # Update bandit with reward
                ctx = torch.tensor(metrics["profile"])
                r = GAMMA_REWARD * metrics["delta_perf"] - DELTA_KB * metrics.get("kb", 0)
                bandit.update(ctx, r, server_round)
        
        return None, {}
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        return []
    
    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

if __name__ == "__main__":
    client_resources = {"num_gpus": 0.25}  # Share GPU among 4 clients
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=NoPrivacyStrategy(),
        client_resources=client_resources
    )