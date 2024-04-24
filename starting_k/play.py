from glob import glob
from os.path import join as pjoin

import textworld.gym
from textworld import EnvInfos

GAMES_PATH = "sample_games"
gamefiles = glob(pjoin(GAMES_PATH, "*.ulx"))
print(f"Found {len(gamefiles)} games.")

gamefile = gamefiles[0]

requested_infos = EnvInfos(description=True, inventory=True, extras=["recipe", "walkthrough"])
env_id = textworld.gym.register_games([gamefile], requested_infos)

env = textworld.gym.make(env_id)

obs, infos = env.reset()

print(f"Walkthrough: {'. '.join(infos['extra.walkthrough'])}")
print(infos["extra.recipe"])

env.render()

score = 0
done = False
while not done:
    command = input("> ")
    obs, score, done, infos = env.step(command)
    env.render()

# game = textworld.Game.load(gamefile.replace(".ulx", ".json"))
# textworld.render.visualize(game)
