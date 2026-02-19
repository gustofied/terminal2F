## GPU Nodes and Shapes

A node is one machine. A GPU is one accelerator inside a machine. These are different "shapes":

**1 node, 1 GPU** - simplest. You can serve or train small jobs. If you try to both serve and train, they'll fight for the same GPU (memory/compute), so it's usually painful unless everything is tiny.

**1 node, 2 GPUs** - still one machine, but you can separate roles by GPU. This is the common pattern for RL/RFT setups:
- Inference server pinned to `CUDA_VISIBLE_DEVICES=0`
- Trainer/workers on `CUDA_VISIBLE_DEVICES=1`
- This is what people mean by "need 2 GPUs" - not two nodes, just two GPUs available on the same box.

**2 nodes, 1 GPU each** - distributed across machines. You need networking, orchestration, and distributed training comms (NCCL etc). More complex and only worth it if you can't fit the work on one box or you need to scale.

**2 nodes, 2+ GPUs each** - full multi-node cluster. Best for scaling up, not for starting.

### SSH Key Auth

Useful videos:
- [SSH Public Key Authentication Oversimplified](https://www.youtube.com/watch?v=j4J_MxkY-4s)
- [SSH Keys](https://www.youtube.com/watch?v=dPAw4opzN9g)

You generate the keypair on your machine because the private key must never leave it. If the server/provider generated it, they could keep a copy, which defeats the point.

**How it works:**
1. You upload the public key to the server (goes into `~/.ssh/authorized_keys`)
2. When you connect, the server says "prove you have the matching private key"
3. Your laptop signs a one-time challenge with the private key
4. The server verifies the signature with the public key. If it checks out, you're in.

It's not encryption - it's proof of identity. The public key is safe to share. Adding a passphrase protects you if your laptop is stolen.

### Getting on a GPU Node

1. Pick a single-node GPU instance with 2x GPUs (one machine, two GPUs)
2. Generate a dedicated SSH keypair on your Mac:
   ```bash
   ssh-keygen -t ed25519 -a 64 -f ~/.ssh/primeintellect_ed25519 -C "primeintellect"
   ```
3. Copy the public key: `cat ~/.ssh/primeintellect_ed25519.pub`
4. In the provider dashboard, add a new SSH key and paste the public key
5. Deploy the instance
6. SSH in:
   ```bash
   ssh -i ~/.ssh/primeintellect_ed25519 -p <port> root@<ip>
   ```
7. Verify GPUs: `nvidia-smi` (should show GPU 0 and GPU 1)
8. Check storage: `df -h /workspace` (network mount) and `df -h /` (local disk)

### Gotchas from First Run

- `/workspace` is a network mount on RunPod; local disk is `/`
- `prime env install` downloads environments from the hub; private envs need `prime login` on the node
- `prime` and `vf-eval` installed via `uv tool` end up in isolated tool envs - they can't see each other's packages. Fix: create one project venv and install everything there.
