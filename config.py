test = dict(
        #max step over the episodes
        max_step = 40,
        #actor pickly location
        actor = './model/sculpt-run/actor.pkl',
        #test image
        img = './image/test.png'
)



# parser = argparse.ArgumentParser(description='Learning to Paint')
# parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
# parser.add_argument('--actor', default='./model/Paint-run1/actor.pkl', type=str, help='Actor model')
# parser.add_argument('--renderer', default='./renderer.pkl', type=str, help='renderer model')
# parser.add_argument('--img', default='image/test.png', type=str, help='test image')
# parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
# parser.add_argument('--divide', default=4, type=int, help='divide the target image to get better resolution')
# args = parser.parse_args()

train = dict(
        #training warmup training number
        warmup = 400,
        #discount factor
        discount = 0.95**5,
        #minibatch size
        batch_size = 96,
        #replay memory size
        rmsize = 800,
        #concurrent environment size
        env_batch = 96,
        #moving average for target network
        tau = 0.001,
        #max length for episode
        max_step = 40,
        #noise level for parameter space noise
        noise_factor = 0,
        #how many episodes to perform for validation
        validate_interval = 50,
        #how many episodes to perform for validation
        validate_episodes = 5,
        #total train times
        train_times = 2000000,
        #train times for each episode
        episode_train_times = 10,
        #resuming model path for testing
        resume = None,
        #resuming model path for testing
        output = './model',
        #print out helpful info 
        debug = True,
        #random seed
        seed = 1234
)

path = dict(
        root = 'c:/Users/Stephen Kellett/Documents/Project/Project Code',
        #resuming model path for testing
        output = 'c:/Users/Stephen Kellett/Documents/Project/Project Code/model'
)

# hyper-parameter
    # parser.add_argument('--warmup', default=400, type=int, help='timestep without training but only filling the replay memory')
    # parser.add_argument('--discount', default=0.95**5, type=float, help='discount factor')
    # parser.add_argument('--batch_size', default=96, type=int, help='minibatch size')
    # parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
    # parser.add_argument('--env_batch', default=96, type=int, help='concurrent environment number')
    # parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    # parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
    # # parser.add_argument('--noise_factor', default=0, type=float, help='noise level for parameter space noise')
    # parser.add_argument('--validate_interval', default=50, type=int, help='how many episodes to perform a validation')
    # parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validation')
    # parser.add_argument('--train_times', default=2000000, type=int, help='total traintimes')
    # parser.add_argument('--episode_train_times', default=10, type=int, help='train times for each episode')    
    # parser.add_argument('--resume', default=None, type=str, help='Resuming model path for testing')
    # # parser.add_argument('--output', default='./model', type=str, help='Resuming model path for testing')
    # parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
    # parser.add_argument('--seed', default=1234, type=int, help='random seed')