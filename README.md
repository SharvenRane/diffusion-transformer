# diffusion-transformer

A compact, readable implementation of the Diffusion Transformer (DiT). It replaces the convolutional UNet that usually backs a diffusion model with a plain transformer that operates on image patches. The conditioning on diffusion timestep and class label is injected through adaptive layer norm, the adaLN-Zero variant from the DiT paper.

The whole thing is small on purpose. The default config builds a 16x16 single channel image model with a hidden size of 48 and two transformer blocks, so every test runs on CPU in a couple of seconds with no downloads.

## What is in here

The pipeline mirrors the original DiT.

1. **Patchify.** A strided convolution cuts the image into non overlapping patches and projects each one to the hidden dimension. A fixed 2D sine cosine positional embedding is added so the transformer knows where each patch sat.
2. **Conditioning.** The scalar timestep is turned into a vector with a sinusoidal frequency table and a small MLP. The class label goes through an embedding table. The two are summed into one conditioning vector.
3. **Transformer blocks.** Each block runs self attention and an MLP. Before each sublayer the tokens pass through a layer norm with no learned affine, and the conditioning vector supplies the shift, scale, and gate. This is the adaLN-Zero scheme: the modulation layers start at zero so a fresh block is the identity function, which makes early training stable.
4. **Unpatchify.** A final adaLN modulated layer projects each token back to a patch of pixels, and the patches are reassembled into the output image.

The model predicts noise, so the forward pass returns a tensor with the same channel and spatial shape as the input.

## Files

- `src/config.py` is the tiny dataclass config.
- `src/model.py` holds the patch embedder, timestep and label embedders, the DiT block, the final layer, and the full `DiT` module.
- `src/diffusion.py` is a minimal DDPM forward noising process and a noise prediction loss, enough to train a step.
- `tests/test_dit.py` is the behaviour test suite.

## Running

Install the dependencies and run the tests.

```
pip install -r requirements.txt
python -m pytest tests/ -q
```

## What the tests check

These are property and behaviour checks rather than fixed number comparisons.

- The forward pass returns the input image shape.
- Patchify produces the right token count, and unpatchify reassembles the correct spatial shape.
- Changing the timestep embedding changes the output, and so does changing the class label, which confirms the adaLN conditioning path is live. A fresh adaLN-Zero model emits zeros by design, so these tests perturb the modulation weights first to exercise conditioning.
- A freshly built block is the identity at initialization, the property that adaLN-Zero is meant to give.
- The model trains: gradients reach the patch embedder, parameters move after optimizer steps, and the loss on a fixed single batch goes down when you overfit it.

On a CPU run the full suite reports 9 passed in roughly 2 seconds.

## Notes

The config exposes a `learn_sigma` flag. When off the model predicts only noise and the output channel count matches the input. When on it doubles the output channels to also predict a variance, matching the original DiT setup. The tests run with it off.
