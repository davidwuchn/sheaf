# NanoGPT

A port of [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) in Sheaf.
The architecture follows the original faithfully, keeping the same layer structure, weight layout, and numerical results.

The weights for the character-level Shakespeare were converted from the original PyTorch format to SafeTensors.

## Char-level (Shakespeare)

The default configuration is a 10.65M parameter GPT-2 trained on the complete
works of Shakespeare at character level. Pre-trained weights and training data
are included.

Generate text:

```
sheaf sample.shf
```

Fine-tune for 100 steps with Adam optimizer:

```
sheaf train.shf
```
