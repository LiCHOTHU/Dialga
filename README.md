# ⏱️ DIALGA: Discrete Integration And Lagrangian Generative Architecture

## Repository Status

The repo is currently transitioning from the earlier latent-video VSFM prototype to an object-state dynamics pipeline.

- New object-state code lives under `src/dynamics/`, `src/data/clevrer_states.py`, and `scripts/train_dynamics.py`.
- Runtime outputs and future checkpoints should be written under `$SCRATCH/Dialga` rather than the repo directory.
- The older latent-space training path is still present as legacy code while the rewrite is being validated.

**DIALGA** is a physics-grounded forward dynamics model that implements **Variational Solver Flow Matching (VSFM)**. 

Standard video generation models treat physical evolution as an arbitrary, conditional visual mapping—leading to "case-based generalization" where models simply hallucinate pixels that look like their training data, breaking the laws of physics in out-of-distribution scenarios. 

DIALGA solves this by interpreting next-state prediction as the numerical solution of a learned **discrete Euler-Lagrange (DEL) equation**. It uses flow matching not as a generic image generator, but as an amortized numerical solver to descend the energy gradient of a learned physical law.

## 🧠 Core Architecture

Unlike standard diffusion models, DIALGA separates the "physics engine" from the "visual generator." 

1. **The Latent Encoder:** Compresses high-dimensional video observations ($o_t$) into a continuous latent state ($q_t$).
2. **The Discrete Lagrangian Network:** A scalar-valued neural network $L_{d,\theta}$ that learns the discrete action (energy) between adjacent states, eliminating the need to manually program mass, friction, or gravity.
3. **The Root-Finding Flow Solver:** A generative flow-matching ODE that transports a noisy candidate state down the energy gradient of the DEL residual until it reaches $R_\theta(y; \xi_t) = 0$. 

---

## 🚀 Installation & Setup

### Environment Requirements
Due to the calculation of the DEL vector field via the energy gradient, DIALGA requires higher-order automatic differentiation (`torch.autograd.grad` with `create_graph=True`). A high-VRAM GPU is strongly recommended.

```bash
git clone [https://github.com/your-username/DIALGA.git](https://github.com/your-username/DIALGA.git)
cd DIALGA
pip install -r requirements.txt
```

### Dataset Preparation (Phase 1: CLEVRER)
We are initially validating the physics engine using the **CLEVRER** dataset (pure elastic collisions, no action conditioning). 

A custom shell script is provided to pull the dataset directly to your HPC storage partition. 

1. Make the download script executable:
```bash
chmod +x scripts/download_clevrer.sh
```
2. Execute the script (default target directory: `/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER`):
```bash
./scripts/download_clevrer.sh
```
You can override the destination with `BASE_DIR=/your/path ./scripts/download_clevrer.sh`.
3. Run the frame extraction utility to convert the raw `.mp4` training files into structured $(o_{t-1}, o_t, o_{t+1})$ triplets for the PyTorch Dataloader:
```bash
python data/extract_frames.py --input /storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video --output /storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/frames
```

---

## 🗺️ Project Roadmap

Testing a fundamentally new mathematical architecture requires a progressive pipeline to isolate solver errors from visual noise.

- [x] **Phase 0: Mathematical Formulation:** Validate the continuous-to-discrete variational mechanics chain.
- [ ] **Phase 1: Pure Physics Sandbox (CLEVRER):** Prove that the amortized flow-matching solver can learn a Lagrangian energy landscape and perfectly conserve momentum/energy in out-of-distribution collisions.
- [ ] **Phase 2: Action-Conditioned Dynamics (BAIR Robot Pushing):** Introduce the control input ($a_t$) to the discrete Lagrangian to test dynamics-aware latent transitions under human/robotic command.
- [ ] **Phase 3: Complex Manipulation (Bridge V2):** Scale the model to real-world, high-occlusion environments featuring complex grasping and dynamic physical constraints.

---

## 💻 Quickstart: Training Loop

The training objective consists of three coupled losses: Reconstruction ($\mathcal{L}_{rec}$), Discrete Variational Consistency ($\mathcal{L}_{DEL}$), and the Flow-Matching Solver Loss ($\mathcal{L}_{VSFM}$).

```python
from dialga.models import VideoEncoder, VideoDecoder, DiscreteLagrangianNet
from dialga.solver import calculate_DEL_residual

# Initialize core architecture
encoder = VideoEncoder()
decoder = VideoDecoder()
lagrangian = DiscreteLagrangianNet()

# The solver field is generated dynamically via autograd
# solver_field = -1 * torch.autograd.grad(outputs=energy, inputs=y_s, create_graph=True)[0]
```
*(See `train.py` for the full training loop and hyperparameter configurations.)*

---

## 📝 Citation & Author

**Author:** Liquan Wang (Licho)

If you find this code useful in your research, please consider citing the foundational VSFM framework:
```bibtex
@article{vsfm2026,
  title={Variational Solver Flow Matching for Forward Dynamics Generation},
  author={Wang, Liquan},
  journal={TBD},
  year={2026}
}
```
