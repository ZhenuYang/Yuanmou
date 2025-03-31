#### Train Learngene

Directly extract the learngene from the pretrained large model, therefore omitting this step.

#### Extract Learngene

First, train the auxiliary models to help build the learngene pool

For example, the command to train the auxiliary models with 6 blocks.

```python
python distill.py --config configs/conf_aux.yaml
```

The default parameters of the experiment are shown in `configs/conf_aux.yaml`.

If we need auxiliary models with 9 blocks, we only need to make change to 'student_model' .

Then we can build the learngene pool and finetune it.

```python
python main.py --config configs/conf_build.yaml
```

The default parameters of the experiment are shown in `configs/conf_build.yaml`.

\# init_stitch_mode = ours or snnet, 'ours' means to initialize the stitching layers by the proposed method and snnet means by Sn-Net

#### Initialize Learngene

In this part, we can build variable-sized models from the learngene pool.

```python
python main.py --config configs/conf_ini.yaml
```

The default parameters of the experiment are shown in `configs/conf_ini.yaml`.

To build Learngenepool and descendant models of different sizes, you only need to modify some hyperparameters.
