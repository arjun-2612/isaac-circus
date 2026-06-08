"""Wrapper for recording videos."""
from gymnasium.wrappers import RecordVideo

class MultiFrameRecordVideo(RecordVideo):
    """RecordVideo wrapper that handles multiple frames per step."""
    
    def step(self, action):
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)

        if not (self.terminated or self.truncated):
            self.step_id += 1
            if not self.is_vector_env:
                if terminateds or truncateds:
                    self.episode_id += 1
                    self.terminated = terminateds
                    self.truncated = truncateds
            elif terminateds[0] or truncateds[0]:
                self.episode_id += 1
                self.terminated = terminateds[0]
                self.truncated = truncateds[0]

            if self.recording:
                assert self.video_recorder is not None
                frames = self.env.render()
                
                if frames is not None:
                    if isinstance(frames, list):
                        for frame in frames:
                            self.video_recorder.recorded_frames.append(frame)
                            self.recorded_frames += 1
                    else:
                        self.video_recorder.recorded_frames.append(frames)
                        self.recorded_frames += 1
                
                if self.video_length > 0:
                    if self.recorded_frames > self.video_length:
                        self.close_video_recorder()
                else:
                    if not self.is_vector_env:
                        if terminateds or truncateds:
                            self.close_video_recorder()
                    elif terminateds[0] or truncateds[0]:
                        self.close_video_recorder()

            elif self._video_enabled():
                self.start_video_recorder()

        return observations, rewards, terminateds, truncateds, infos

    def reset(self, **kwargs):
        observations = super(RecordVideo, self).reset(**kwargs)
        self.terminated = False
        self.truncated = False
        
        if self.recording:
            assert self.video_recorder is not None
            self.video_recorder.recorded_frames = []
            frames = self.env.render()
            
            if frames is not None:
                if isinstance(frames, list):
                    for frame in frames:
                        self.video_recorder.recorded_frames.append(frame)
                        self.recorded_frames += 1
                else:
                    self.video_recorder.recorded_frames.append(frames)
                    self.recorded_frames += 1
            
            if self.video_length > 0 and self.recorded_frames > self.video_length:
                self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()
        
        return observations