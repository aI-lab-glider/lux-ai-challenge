import json

def parse_kaggle_env_episode(episode):
    observations = []
    for i, step in enumerate(episode['steps']):
        obs = json.loads(step[0]['observation']['obs'])
        if i > 0:
            board = observations[-1]['board'].copy()
            board_state = obs['board']
            for key in [
                'rubble',
                'lichen',
                'lichen_strains'
            ]:
                for k, v in board_state[key].items():
                    k = k.split(',')
                    x, y = int(k[0]), int(k[1])
                    board[x, y] = v
            obs['board'] = board

        
        observations.append(obs)
        
    return observations