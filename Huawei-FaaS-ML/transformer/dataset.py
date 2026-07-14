import torch


from feature_engineering.sequence_builder import HuaweiSequenceDataset

class KashifDataset(HuaweiSequenceDataset):

    def __getitem__(self, idx):

        sample = super().__getitem__(idx)

        x = sample["x"]

        y = sample["y"]

        # --------------------------------------
        # Time Features
        # --------------------------------------

        hour_sin = x[:, -4].unsqueeze(1)
        hour_cos = x[:, -3].unsqueeze(1)

        minute_sin = x[:, -2].unsqueeze(1)
        minute_cos = x[:, -1].unsqueeze(1)

        past_time = torch.cat(
            [
                hour_sin,
                hour_cos,
                minute_sin,
                minute_cos
            ],
            dim=1
        )

        # --------------------------------------
        # Future Time Features
        # --------------------------------------

        future_time = torch.zeros(
            (
                len(y),
                4
            ),
            dtype=torch.float32
        )

        future_time[:,0] = hour_sin[-1]
        future_time[:,1] = hour_cos[-1]
        future_time[:,2] = minute_sin[-1]
        future_time[:,3] = minute_cos[-1]

        # --------------------------------------

        return {

            "past_values": x,

            "past_time_features": past_time,

            "future_time_features": future_time,

            "static_features": {

                "category": sample["category"],

                "stability": sample["stability"]

            },

            "target": y

        }
