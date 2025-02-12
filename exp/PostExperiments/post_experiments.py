import torch
import os
from exp import g_exp

def load_model(wrapper, model_path, device='auto'):
    """
    Loads the SDAC model from the specified path.

    Args:
        wrapper (Wrapper): An instance of the Wrapper class with correct state_dim, action_dim, and action_atoms.
        model_path (str): Path to the saved model.pth file.
        device (str): Device to load the model onto ('cuda', 'cpu', or 'auto' for automatic selection).

    Returns:
        SDAC: An instance of the SDAC class with loaded model parameters.
    """
    # Determine the device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Initialize the SDAC algorithm linked to the environment wrapper
    algo = wrapper.sdac()
    algo.device = device

    # Build the neural networks to initialize their structure
    algo.build_network()

    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Load state dictionaries into each component
    algo.qf1.load_state_dict(checkpoint['qf1'])
    algo.qf2.load_state_dict(checkpoint['qf2'])
    algo.actor.load_state_dict(checkpoint['actor'])
    algo.target_qf1.load_state_dict(checkpoint['target_qf1'])
    algo.target_qf2.load_state_dict(checkpoint['target_qf2'])
    algo.target_actor.load_state_dict(checkpoint['target_actor'])
    algo.best_actor.load_state_dict(checkpoint['best_actor'])

    # Load observation normalization parameters if available
    model_dir = os.path.dirname(model_path)
    obs_norm_path = os.path.join(model_dir, 'obs-norm.pth')
    if os.path.exists(obs_norm_path):
        obs_norm = torch.load(obs_norm_path, map_location='cpu')
        # Convert to numpy arrays and set observation normalization
        algo.obervation_apply = (
            obs_norm[0].cpu().numpy(),
            obs_norm[1].cpu().numpy(),
            obs_norm[2].cpu().numpy()
        )

    return algo

# Example usage:
# Assuming you have an initialized Wrapper instance with correct parameters:
# wrapper = Wrapper()
# wrapper.state_dim = 10  # Set according to your environment
# wrapper.action_dim = 4
# wrapper.action_atoms = 51
# model_path = "path/to/model.pth"
# algo = load_model(wrapper, model_path)

def get_action(algo, state):
    with torch.no_grad():
        state_tensor = torch.tensor(state).float().unsqueeze(0).to(algo.device)
        if algo.observation_norm and algo.obervation_apply is not None:
            state_tensor = algo.apply_observation_norm(state_tensor)
        logits = algo.best_actor(state_tensor)
        action = algo.best_actor.eval(logits)
        return action.squeeze().cpu().numpy()

if __name__=='__main__':
    # path = './exp/'
    # dir_list = os.listdir(path)
    # print("Files and directories in '", path, "' :")
    # print(dir_list)
    g_exp[args.exp]
    model_path = './exp/PostExperiments/model.py'
    # Load saved checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    load_model(wrapper, model_path, device='auto')

    # Load best actor weights
    actor.load_state_dict(checkpoint['best_actor'])