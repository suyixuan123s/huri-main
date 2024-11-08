from huri.learning.method.AlphaZero.game.tubeswap import Game
if __name__ == "__main__":
    env = Game()
    for _ in range(10):
        print(env.reset())