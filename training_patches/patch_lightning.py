"""
Patch piper_train/vits/lightning.py for pytorch-lightning 2.x compatibility.

PL 2.x removed automatic optimization with multiple optimizers / optimizer_idx,
which VITS GAN training (separate generator/discriminator optimizers) relies on.
This switches the model to manual optimization.

Usage: python3 patch_lightning.py /path/to/piper/src/python/piper_train/vits/lightning.py
"""
import sys

path = sys.argv[1]
src = open(path).read()

src = src.replace(
    "        self._y_hat = None\n",
    "        self._y_hat = None\n        self.automatic_optimization = False\n",
    1,
)

old_step = '''    def training_step(self, batch: Batch, batch_idx: int, optimizer_idx: int):
        if optimizer_idx == 0:
            return self.training_step_g(batch)

        if optimizer_idx == 1:
            return self.training_step_d(batch)'''

new_step = '''    def training_step(self, batch: Batch, batch_idx: int):
        opt_g, opt_d = self.optimizers()

        loss_gen_all = self.training_step_g(batch)
        opt_g.zero_grad()
        self.manual_backward(loss_gen_all)
        opt_g.step()

        loss_disc_all = self.training_step_d(batch)
        opt_d.zero_grad()
        self.manual_backward(loss_disc_all)
        opt_d.step()

    def on_train_epoch_end(self):
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()'''

assert old_step in src, "training_step block not found - piper_train source may have changed"
src = src.replace(old_step, new_step)

open(path, "w").write(src)
print("patched", path)
