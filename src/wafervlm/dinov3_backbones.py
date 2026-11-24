"""Identities from the official LVD-1689M Hub revisions (Git config blob and LFS payload)."""
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Backbone:
    model_id: str
    revision: str
    weights_sha256: str
    config_blob: str
    hidden_size: int
    parameters: int

    def verify_file(self, name, path):
        path = Path(path)
        if name == "model.safetensors":
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != self.weights_sha256:
                raise ValueError("Official weight checksum mismatch")
            return digest
        if name != "config.json":
            raise ValueError("Unexpected backbone file")
        raw = path.read_bytes()
        if hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest() != self.config_blob:
            raise ValueError("Official configuration blob mismatch")
        return hashlib.sha256(raw).hexdigest()


BACKBONES = {
    "small": Backbone("facebook/dinov3-vits16-pretrain-lvd1689m",
                      "114c1379950215c8b35dfcd4e90a5c251dde0d32",
                      "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d",
                      "fdff48569812f3cbadf253762565c4ce9dacddac", 384, 21596544),
    "base": Backbone("facebook/dinov3-vitb16-pretrain-lvd1689m",
                     "5931719e67bbdb9737e363e781fb0c67687896bc",
                     "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
                     "9f0c03c12a21df9678269d96dac7b59819349792", 768, 85660416),
}
