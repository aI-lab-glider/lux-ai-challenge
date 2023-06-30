import agents.tournament as tournament
from pathlib import Path
import sys
import subprocess
import datetime
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

if __name__ == '__main__':
    agents = [Path(p).stem for p in sys.argv[1:]]
    players: list[Path] = []

    template_dir = Path(tournament.__file__).parent
    for agent_module in agents:
        with open(template_dir / 'player_template', 'r') as f:
            template = f.read()
            template = template.replace('{{module}}', agent_module)
            player = Path(f'{agent_module}.py')
            with open(player, 'w') as fp:
                fp.write(template)
                players.append(player)

    tournament_logs = Path('tournament_logs') / \
        f'{datetime.datetime.now()}_{"_".join(agents)}'

    for file in tournament_logs.parent.iterdir():
        file.unlink()

    result = subprocess.run(['luxai-s2', players[0], players[1], '-v', '3',
                            '-s', '101', '-o', f'{tournament_logs}.json',])

    # for player in players:
    #     if player.exists():
    #         player.unlink()
