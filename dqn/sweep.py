from agent import main
import wandb

def new_main():
    main('sweep')

wandb.agent('ac3dhk70', new_main)
