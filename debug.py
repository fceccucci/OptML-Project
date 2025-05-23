from flwr.simulation import run_simulation
from src.server_app import app as server_app
from src.client_app import app as client_app

run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=4,
)