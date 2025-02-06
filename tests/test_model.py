import unittest

import torch

from model import SuperCool, EncoderBlock, SubpixelConv2d, Bouncer, DetectorBlock


class TestSuperCool(unittest.TestCase):
    def test_initialization(self):
        model = SuperCool("bilinear", 2, 64, 2, 3)
        self.assertIsInstance(model, SuperCool)

    def test_invalid_base_upscaler(self):
        with self.assertRaises(ValueError):
            SuperCool("invalid", 2, 64, 2, 3)

    def test_invalid_upscale_ratio(self):
        with self.assertRaises(ValueError):
            SuperCool("bilinear", 3, 64, 2, 3)

    def test_invalid_num_channels(self):
        with self.assertRaises(ValueError):
            SuperCool("bilinear", 2, 0, 2, 3)

    def test_invalid_hidden_ratio(self):
        with self.assertRaises(ValueError):
            SuperCool("bilinear", 2, 64, 3, 3)

    def test_invalid_num_layers(self):
        with self.assertRaises(ValueError):
            SuperCool("bilinear", 2, 64, 2, 0)

    def test_forward(self):
        model = SuperCool("bilinear", 2, 64, 2, 3)
        x = torch.randn(1, 3, 64, 64)
        y = model(x)
        self.assertEqual(y.shape, (1, 3, 128, 128))


class TestEncoderBlock(unittest.TestCase):
    def test_initialization(self):
        block = EncoderBlock(64, 2)
        self.assertIsInstance(block, EncoderBlock)

    def test_invalid_hidden_ratio(self):
        with self.assertRaises(ValueError):
            EncoderBlock(64, 3)

    def test_forward(self):
        block = EncoderBlock(64, 2)
        x = torch.randn(1, 64, 32, 32)
        y = block(x)
        self.assertEqual(y.shape, (1, 64, 32, 32))


class TestSubpixelConv2d(unittest.TestCase):
    def test_initialization(self):
        layer = SubpixelConv2d(64, 2, 3, 1)
        self.assertIsInstance(layer, SubpixelConv2d)

    def test_forward(self):
        layer = SubpixelConv2d(64, 2, 3, 1)
        x = torch.randn(1, 64, 32, 32)
        y = layer(x)
        self.assertEqual(y.shape, (1, 12, 32, 32))


class TestBouncer(unittest.TestCase):
    def test_initialization(self):
        model = Bouncer("small")
        self.assertIsInstance(model, Bouncer)

    def test_invalid_model_size(self):
        with self.assertRaises(ValueError):
            Bouncer("invalid")

    def test_forward(self):
        model = Bouncer("small")
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        self.assertEqual(y.shape, (1, 1))


class TestDetectorBlock(unittest.TestCase):
    def test_initialization(self):
        block = DetectorBlock(64, 64, 256)
        self.assertIsInstance(block, DetectorBlock)

    def test_forward(self):
        block = DetectorBlock(64, 64, 256)
        x = torch.randn(1, 64, 32, 32)
        y = block(x)
        self.assertEqual(y.shape, (1, 256, 32, 32))


if __name__ == "__main__":
    unittest.main()
