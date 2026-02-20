import flwr as fl, logging, numpy as np, torch
from src.globals import *
from src.data_utils import load_split
from src.client import KnexaClient
import flwr.common as fl_common

torch.backends.cuda.matmul.allow_tf32 = True
 
logging.basicConfig(filename="experimental_artifacts/baselines/others/logs/local_only.log", level=logging.DEBUG)

CLIENT_SPLITS, GLOBAL_TEST = load_split(NUM_CLIENTS)
 
def client_fn(cid: str):
    cid = int(cid)
    train, val = CLIENT_SPLITS[cid]
    return KnexaClient(cid, train, val)
 
class LocalOnlyStrategy(fl.server.strategy.Strategy):
    def initialize_parameters(self, client_manager):
        """Initialize with dummy parameters - local training doesn't need global parameters"""
        return fl_common.ndarrays_to_parameters([np.array([0.0])])
    
    def evaluate(self, server_round, parameters):
        return None
        
    def configure_fit(self, server_round, parameters, client_manager):
        clients = client_manager.sample(num_clients=NUM_CLIENTS, min_num_clients=NUM_CLIENTS)
        config = {"round": server_round}  # No role_config - local training only
        # Pass dummy parameters - local training uses own model parameters
        dummy_params = fl_common.ndarrays_to_parameters([np.array([0.0])])
        return [(client, fl_common.FitIns(dummy_params, config)) for client in clients]
 
    def aggregate_fit(self, server_round, results, failures):
        return None, {}
    
    def configure_evaluate(self, server_round, parameters, client_manager):
        """Select clients for evaluation to track progress"""
        clients = client_manager.sample(num_clients=NUM_CLIENTS, min_num_clients=NUM_CLIENTS)
        dummy_params = fl_common.ndarrays_to_parameters([np.array([0.0])])
        return [(client, fl_common.EvaluateIns(dummy_params, {"round": server_round})) for client in clients]
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results from all clients"""
        if not results:
            return None, {}
            
        # Collect evaluation metrics
        pass_at_1_scores = []
        client_results = {}
        
        for client_proxy, eval_res in results:
            metrics = eval_res.metrics
            client_id = metrics.get("client_id", "unknown")
            pass_at_1 = metrics.get("pass@1", 0.0)
            
            pass_at_1_scores.append(pass_at_1)
            client_results[f"client_{client_id}"] = pass_at_1
            
            logging.info(f"Round {server_round} - Client {client_id}: pass@1={pass_at_1:.3f}")
        
        # Calculate aggregate metrics
        if pass_at_1_scores:
            avg_pass_at_1 = sum(pass_at_1_scores) / len(pass_at_1_scores)
            
            aggregate_metrics = {
                "avg_pass@1": avg_pass_at_1,
                "num_clients": len(pass_at_1_scores),
                **client_results
            }
            
            logging.info(f"Round {server_round} - Average pass@1: {avg_pass_at_1:.3f}")
            return avg_pass_at_1, aggregate_metrics
        
        return None, {}

if __name__ == "__main__":
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=LocalOnlyStrategy()
    )