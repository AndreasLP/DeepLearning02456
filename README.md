Code for project 2D on key point learning in RL.

The code for training networks on the HPC is located in three directories; splitting the code up in three
segments.

A replaybuffer is needed when training the IMM model on its own. One was provided by
Nicklas Hansen at
https://onedrive.live.com/?authkey=%21AJFfJ76wFCR%2DlMs&id=FC48E2EBC63C9E7A%21768516&cid=FC48E2EBC63C9E7A
(last access 03/01/2021). Code for training the IMM model on its own is setup to use a replay buffer named 
cartpole_swingup_40000_50000.pt.

Note the SAC+AE uses the RAD model i.e. the observations are padded and then randomly cropped back to the original size.

1. key_point_learning 
    The code here is used for training IMM models using either IMM 3 or IMM 9.
    The former uses a single RGB image and outputs three key points while the
    latter uses three RGB images and outputs nine key points
2. Poster_networks
    Code used for generating results seen in the poster. Each subdirectory is
    selfcontained and has a SAC model which may be extended.
        1. pytorch_sac_ae_before_keynet:
            Contains the RAD model used to replicate results
        2. pytorch_sac_ae_modified:
            Contains the RAD model with a IMM 9 with shared image encoder
            between KeyNet, Actor, and Critic
        3. pytorch_sac_ae_only_keypoints:
            Contains the SAC model where the AE is removed but IMM 9 is added.
            KeyNet's image encoder is updated when training the actor and the
            critic
3. Parallel_networks
    Code used for generating results where the KeyNet's image encoder is not
    updated during training of Actor or Critic. sac_keypoints_NUM is similar to
    pytorch_sac_ae_only_keypoints with the change that the encoder is not
    updated as much - NUM specifies whether IMM 3 or IMM 9 is used. pretrained
    marks the folders where the same is done, but, the IMM model is pretrained.
    sac_zero_input is the SAC model where all observations are zeroed s.t. they
    do not provide information for the model - used as a baseline.
    SAC+AE+KeyNet9 is the RAD model with IMM 9 as in 2.2, but, the image
    encoder is not updated when training the Actor and the Critic. The IMM
    model is also pretrained.
	
The remaining files and folders in the repository will be described in the following:

Folders:
	Checkpoints:
	 Contains fully trained .pt model files of various IMM models used for plotting
	 in "self_supervised_keypoint_learning_results.ipynb".
	Logs:
	 Contains evaluation and training log files from RL models. The log files are used
	 to extract episode rewards and various losses such as IMM losses for plotting in
	 "self_supervised_keypoint_learning_results.ipynb".
Files:
	self_supervised_keypoint_learning_results.ipynb:
	 This is our primary notebook which showcases most results from our poster and all
	 results from our report + some extra results. The only thing required to run it is
	 to have the cartpole_swingup_40000_50000.pt file mentioned previously located in
	 the same folder as the notebook.
	colab_keypoint_learning.ipynb:
	 Colab notebook used initially in our project and for most of our poster plots.
	 The notebook requires a certain folder structure and specific files to run which have
	 not been setup in this repository since this notebook is not our primary notebook.
	 Refer to "self_supervised_keypoint_learning_results.ipynb" for our primary notebook.
	Poster.pdf:
	 Poster of our project used at the poster session.
	Project synopsis.pdf:
	 Synopsis of our project.
	 Refer to this file for a short introduction to what our project is about.














