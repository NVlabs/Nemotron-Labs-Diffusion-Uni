import unittest
import torch
from corruption import ages_at_time, corrupt


class IdentityTree:
    depth = 17
    def sample_representatives(self, clean, depths, generator=None):
        subtree_size = 1 << (self.depth - depths)
        starts = (clean // subtree_size) * subtree_size
        offsets = torch.randint(1 << self.depth, clean.shape, generator=generator)
        return starts + offsets.remainder(subtree_size)


class CorruptionTest(unittest.TestCase):
    def test_equation_16_boundaries(self):
        starts = torch.tensor([0., .125, .25, .5, .75], dtype=torch.float64)
        ages = ages_at_time(.5, starts, .25, 5)
        self.assertEqual(ages.tolist(), [4, 4, 4, 0, 0])
        self.assertEqual(ages_at_time(.375, starts, .5, 5).tolist(), [3, 2, 1, 0, 0])

    def test_two_levels_is_one_jump(self):
        clean = torch.arange(10000) % (1 << 17)
        s = corrupt(clean, IdentityTree(), num_tree_levels=2, epsilon=.1, time=.5,
                    generator=torch.Generator().manual_seed(7))
        self.assertEqual(s.decoding_steps, 1)
        self.assertEqual(set(s.ages.tolist()), {0, 1})
        self.assertEqual(set(s.depths.tolist()), {0, 17})
        self.assertEqual(set(s.noise.tolist()), {0., 1.})
        self.assertTrue(torch.equal(s.representatives[~s.supervised], clean[~s.supervised]))

    def test_subtree_membership_and_endpoints(self):
        clean = torch.arange(8192) * 13 % (1 << 17)
        for levels in [2, 5, 9, 18]:
            s = corrupt(clean, IdentityTree(), num_tree_levels=levels, epsilon=.6, time=.6)
            shifts = 17 - s.depths
            self.assertTrue(torch.equal(clean >> shifts, s.representatives >> shifts))
            for t, expected in [(0.,17),(1.,0)]:
                end = corrupt(clean, IdentityTree(), num_tree_levels=levels, epsilon=.6, time=t)
                self.assertTrue(end.depths.eq(expected).all())

    def test_structural_tokens_stay_clean(self):
        clean = torch.tensor([0, 500, 131072, 131074])
        structural = clean >= (1 << 17)
        s = corrupt(clean, IdentityTree(), time=1., structural_mask=structural)
        self.assertTrue(torch.equal(s.representatives[structural],clean[structural]))
        self.assertFalse(s.supervised[structural].any())


if __name__ == '__main__':
    unittest.main()
