from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt

from scripts.render_figures import save


class FigureFilenameTests(unittest.TestCase):
    def test_decimal_sample_ids_and_seeds_do_not_overwrite_each_other(self):
        scratch = Path(__file__).resolve().parents[1] / 'build'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            for seed in (307, 1101):
                fig, ax = plt.subplots(); ax.plot([0, 1])
                save(fig, Path(folder) / f'rsrd_20230409020600.400_s{seed}')
            self.assertEqual(len(list(Path(folder).glob('*.png'))), 2)
            self.assertTrue((Path(folder) / 'rsrd_20230409020600.400_s307.pdf').exists())


if __name__ == '__main__':
    unittest.main()
