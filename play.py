from train import *


try:
    checkpoint = K // log_interval
    agent.pio.load_state_dict(torch.load(f"checkpoints/pi{checkpoint}.pt"))
    print(f"using pi{checkpoint}.pt")
except:
    print("using default")


print(f"return: {agent.play()}")
