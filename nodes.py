"""
ComfyUI-AudioBridge
Bridges ComfyUI's native AUDIO type to other audio formats used by
custom nodes (e.g. MuseTalk, VoiceCraft).
"""
import torch
import torchaudio


class AudioToTensor:
    """
    Converts ComfyUI's native AUDIO type to a raw audio tensor.
    
    ComfyUI AUDIO = {"waveform": Tensor, "sample_rate": int}
    Output = mono torch tensor resampled to target rate (default 16kHz).
    
    Compatible with MuseTalk-KJ's VCAUDIOTENSOR input type and any node
    that expects a raw audio waveform tensor.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_sample_rate": ("INT", {
                    "default": 16000,
                    "min": 8000,
                    "max": 48000,
                    "step": 1000,
                    "tooltip": "Target sample rate in Hz. 16000 for Whisper/MuseTalk."
                }),
            },
        }

    RETURN_TYPES = ("VCAUDIOTENSOR", "INT")
    RETURN_NAMES = ("audio_tensor", "audio_dur")
    FUNCTION = "convert"
    CATEGORY = "AudioBridge"
    DESCRIPTION = "Converts ComfyUI AUDIO to a raw tensor (VCAUDIOTENSOR). Use with MuseTalk, VoiceCraft, or any node expecting raw audio tensors."

    def convert(self, audio, target_sample_rate):
        waveform = audio["waveform"].squeeze(0)  # (channels, samples)
        sample_rate = audio["sample_rate"]

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)

        # Resample if needed
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )

        audio_dur = waveform.shape[1] / target_sample_rate
        return (waveform, int(audio_dur))


class TensorToAudio:
    """
    Converts a raw audio tensor back to ComfyUI's native AUDIO type.
    Useful for routing processed audio back into standard ComfyUI nodes.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_tensor": ("VCAUDIOTENSOR",),
                "sample_rate": ("INT", {
                    "default": 16000,
                    "min": 8000,
                    "max": 48000,
                    "step": 1000,
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "convert"
    CATEGORY = "AudioBridge"
    DESCRIPTION = "Converts a raw audio tensor (VCAUDIOTENSOR) back to ComfyUI AUDIO type."

    def convert(self, audio_tensor, sample_rate):
        # Ensure correct shape: (batch, channels, samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        elif audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)

        return ({"waveform": audio_tensor, "sample_rate": sample_rate},)


NODE_CLASS_MAPPINGS = {
    "AudioToTensor": AudioToTensor,
    "TensorToAudio": TensorToAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioToTensor": "Audio → Tensor",
    "TensorToAudio": "Tensor → Audio",
}
