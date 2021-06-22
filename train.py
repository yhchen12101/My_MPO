import parser
import os
import gym
from my_mpo import MPO
import torch
#from dm_control import suite
import dm_control2gym
    
if __name__ == '__main__':
    args = parser.get_parser().parse_args()
    #args = vars(args)  # Converting argparse Namespace to a dict.
    print("Seed:", args.seed)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    #args.device = torch.device("cuda:{}".format(args.device))
    
    save_name = args.save_name
    if args.save_name is None:
        if args.domain is not None:
            args.save_name = args.env + '_' + args.domain + '_' + args.task
            #args.save_name = 'LunarLanderContinuous-v2'  
        else:
            args.save_name = args.env
        args.save_path = os.path.join('./save', args.save_name)
    #args.save_name = 'LunarLanderContinuous-v2'      
    #env = gym.make('LunarLanderContinuous-v2')
    env = dm_control2gym.make(domain_name=args.domain, task_name=args.task)
    #env = suite.load(domain_name="walker",task_name="run", environment_kwargs=dict(flat_observation=True))
    
    model = MPO(args,env)
    model.train()
    
    env.close()
    