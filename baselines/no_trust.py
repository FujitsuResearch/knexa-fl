import flwr as fl, logging, numpy as np, torch
from src.globals import *
from src.data_utils import load_split
from src.client import KnexaClient
from src.bandit import LinUCB
import flwr.common as fl_common

torch.backends.cuda.matmul.allow_tf32 = True
 
logging.basicConfig(filename="experimental_artifacts/baselines/others/logs/no_trust.log", level=logging.DEBUG)

CLIENT_SPLITS, GLOBAL_TEST = load_split(NUM_CLIENTS)
bandit = LinUCB(d=32)
 
def client_fn(cid: str):
    cid = int(cid)
    train, val = CLIENT_SPLITS[cid]
    return KnexaClient(cid, train, val)
 
class NoTrustStrategy(fl.server.strategy.Strategy):
    def __init__(self):
        super().__init__()
        self.client_profiles = {}
    
    def initialize_parameters(self, client_manager):
        return fl_common.ndarrays_to_parameters([np.array([0.0])])
    
    def evaluate(self, server_round, parameters):
        return None
        
    def configure_fit(self, server_round, parameters, client_manager):
        clients = client_manager.sample(num_clients=NUM_CLIENTS, min_num_clients=NUM_CLIENTS)
        
        profiles = []
        for i, client in enumerate(clients):
            profile = self.client_profiles.get(i, np.zeros(16))
            profile[7] = 0.0  # Zero out trust index
            profiles.append(profile)
        
        pairs = bandit.choose_pairs(profiles, NUM_CLIENTS//2, server_round)
        config = {"round": server_round}
        
        for i, j, alpha, T in pairs:
            val_data = list(CLIENT_SPLITS[i][1])
            queries = [ex["prompt"] for ex in val_data[:20]]
            sub_id = server_round % 3
            config[str(i)] = {"role_config": {"role": "student", "teacher_cid": j, "alpha": alpha, "T": T, "queries": queries}}
            config[str(j)] = {"role_config": {"role": "teacher", "queries": queries, "sub_id": sub_id}}
        
        dummy_params = fl_common.ndarrays_to_parameters([np.array([0.0])])
        return [(client, fl_common.FitIns(dummy_params, config)) for client in clients]
 
    def aggregate_fit(self, server_round, results, failures):
        for i, (client_proxy, fit_res) in enumerate(results):
            metrics = fit_res.metrics
            cid = i  # Use index-based client ID mapping
            
            # Reconstruct profile from individual metrics and force trust=0
            profile = [
                metrics.get("profile_0", 0.0), metrics.get("profile_1", 0.0),
                metrics.get("profile_2", 0.0), metrics.get("profile_3", 0.0),
                metrics.get("profile_4", 0.0), metrics.get("profile_5", 0.0),
                metrics.get("profile_6", 0.0), 0.0,  # Force trust to 0
            ]
            self.client_profiles[cid] = np.array(profile + [0.0] * 8)  # Pad to 16
            
            # Update bandit with zero-trust context
            ctx = torch.tensor(np.concatenate([profile + [0.0] * 8, profile + [0.0] * 8]), dtype=torch.float32)
            r = GAMMA_REWARD * metrics["delta_perf"] - DELTA_KB * metrics.get("kb", 0)
            bandit.update(ctx, r, server_round)
        
        return None, {}
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        return []
    
    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

if __name__ == "__main__":
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=NoTrustStrategy(),
        client_resources={"num_gpus": 0.25, "num_cpus": 1}
    )