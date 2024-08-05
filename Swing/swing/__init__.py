from gymnasium.envs.registration import register

register(id='swing-v0',
         entry_point='swing.envs:SwingEnv')