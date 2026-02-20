import flwr as fl, logging, numpy as np, torch, random
from src.globals import *
from src.data_utils import load_split
from src.client import KnexaClient
import flwr.common as fl_common

torch.backends.cuda.matmul.allow_tf32 = True
 
logging.basicConfig(filename="experimental_artifacts/baselines/others/logs/random_pair.log", level=logging.DEBUG)

CLIENT_SPLITS, GLOBAL_TEST = load_split(NUM_CLIENTS)
 
def client_fn(cid: str):
    cid = int(cid)
    train, val = CLIENT_SPLITS[cid]
    return KnexaClient(cid, train, val)
 
class RandomPairStrategy(fl.server.strategy.Strategy):
    def initialize_parameters(self, client_manager):
        return fl_common.ndarrays_to_parameters([np.array([0.0])])
    
    def evaluate(self, server_round, parameters):
        return None
        
    def configure_fit(self, server_round, parameters, client_manager):
        clients = client_manager.sample(num_clients=NUM_CLIENTS, min_num_clients=NUM_CLIENTS)
        
        # Random pairing instead of bandit
        client_ids = list(range(NUM_CLIENTS))
        random.shuffle(client_ids)
        pairs = list(zip(client_ids[::2], client_ids[1::2]))
        
        config = {"round": server_round}
        
        for i, j in pairs:
            alpha = KD_ALPHA_GRID[server_round % len(KD_ALPHA_GRID)]
            T = TEMP_DEFAULT + (server_round % 3) * 0.5
            val_data = list(CLIENT_SPLITS[i][1])
            queries = [ex["prompt"] for ex in val_data[:20]]
            sub_id = server_round % 3
            config[str(i)] = {"role_config": {"role": "student", "teacher_cid": j, "alpha": alpha, "T": T, "queries": queries}}
            config[str(j)] = {"role_config": {"role": "teacher", "queries": queries, "sub_id": sub_id}}
        
        dummy_params = fl_common.ndarrays_to_parameters([np.array([0.0])])
        return [(client, fl_common.FitIns(dummy_params, config)) for client in clients]
 
    def aggregate_fit(self, server_round, results, failures):
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
        strategy=RandomPairStrategy(),
        client_resources={"num_gpus": 0.25, "num_cpus": 1}
    )