from models import *
import argparse
from config_parser import Configuration


parser = argparse.ArgumentParser()
network_choices = ['cohen', 'oksuz']
parser.add_argument('-network', '-n', dest='network', choices=network_choices, type=str.lower, required=True)
parser.add_argument('-path', required=True)  # Path to the model + filename
args = parser.parse_args()

config = Configuration(args.network, "config.ini", args.debug)

checkpoint = torch.load(args.path)

if args.network == 'cohen':
    model = CohenMLP(seq_len=config.seq_len)
elif args.network == 'oksuz':
    model = Oksuz(config.gru, input_size=config.lstm_input_size, hidden_size=config.lstm_hidden_size,
                  seq_len=config.seq_len, num_layers=config.lstm_num_layers,
                  bidirectional=config.lstm_bidirectional)
else:
    import sys  # Should not be able to reach here as we provide a choice.
    print("Invalid network. Exiting...")
    sys.exit(1)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

start, stop, step = 0, 5000, 2
t1 = list(range(start, stop + step, step))

start, stop, step = 0, 2000, 2
t2 = list(range(start, stop + step, step))
