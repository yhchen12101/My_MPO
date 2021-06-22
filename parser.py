import argparse

def get_parser():
    parser = argparse.ArgumentParser("MPO", description="MPO trainer")
    
    # environment related
    parser.add_argument("--env", default="suite", type=str,
                    help="game environment") #paper environment: suite, parkour, ATARI
    parser.add_argument("--domain", default='hopper', type=str,
                    help="domain name (for suite)")
    parser.add_argument("--task", default='stand', type=str,
                    help="task name (for suite)")
    parser.add_argument("--c", default=True, type=bool,
                    help="is it continuous control tasks")
    
    # training related
    parser.add_argument("-pi_hidden", "--policy_hidden", default=100, type=int,
                    help="ploicy net hidden layer")
    parser.add_argument("--q_hidden", default=200, type=int,
                    help="Q function net hidden layer")
    parser.add_argument("-e", "--epsilon", default=0.1, type=float,
                    help="hard constraint for constrained E-step")
    parser.add_argument("-e_mu", "--epsilon_mu", default=0.1, type=float,
                    help="hard constraint of mu for variational distribution")
    parser.add_argument("-e_cov", "--epsilon_cov", default=1e-4, type=float,
                help="hard constraint of covariance for variational distribution")
    parser.add_argument("--discount", default=0.99, type=float,
                help="discount factor")
    parser.add_argument("--lr", default=5e-4, type=float,
                help="learning rate")
    
    
    #device related
    parser.add_argument("--device", default=-1, type=int,
                    help="GPU index to use, for cpu use -1.")
    parser.add_argument("-seed", "--seed", default=1, type=int, nargs="+",
                    help="Random seed.")
    parser.add_argument("-name", "--save_name", default=None, type=str,
                    help="name of the saved model")
    
    return parser