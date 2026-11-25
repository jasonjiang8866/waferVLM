"""Identities from the official LVD-1689M Hub revisions."""
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Backbone:
    model_id: str
    revision: str
    weights_sha256: str
    config_sha256: str
    hidden_size: int
    parameters: int

    def verify_file(self, name, path):
        path = Path(path)
        checksums = {"config.json": self.config_sha256, "model.safetensors": self.weights_sha256}
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != checksums[name]:
            raise ValueError(f"Official {name} checksum mismatch")
        return digest


BACKBONES = {
    "small": Backbone("facebook/dinov3-vits16-pretrain-lvd1689m",
                      "114c1379950215c8b35dfcd4e90a5c251dde0d32",
                      "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d",
                      "9481247be9f95a134a5599402b4bfc838eecdf9a7fffbf4debd1c70ec213898b", 384, 21596544),
    "base": Backbone("facebook/dinov3-vitb16-pretrain-lvd1689m",
                     "5931719e67bbdb9737e363e781fb0c67687896bc",
                     "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
                     "3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec", 768, 85660416),
}
