from __future__ import absolute_import, division, print_function

import os
import tempfile
import unittest

from transformers.tokenizations import bert, utils


class TokenizationTest(unittest.TestCase):

    def test_full_tokenizer(self):
        vocab_tokens = [
            '[UNK]', '[CLS]', '[SEP]', 'want', '##want', '##ed', 'wa', 'un', 'runn', '##ing', ',',
            '[PAD]', '[MASK]'
        ]
        with tempfile.NamedTemporaryFile(delete=False) as vocab_writer:
            vocab_writer.write(''.join([x + '\n' for x in vocab_tokens]).encode('utf-8'))
            vocab_file = vocab_writer.name

        tokenizer = bert.BertTokenizer(vocab_file)
        os.unlink(vocab_file)

        tokens = tokenizer.tokenize(u'UNwant\u00E9d,running')
        self.assertEqual(tokens, ['un', '##want', '##ed', ',', 'runn', '##ing'])

        self.assertEqual(tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

    def test_chinese(self):
        tokenizer = bert.BasicTokenizer()

        self.assertEqual(
            tokenizer.tokenize(u'ah\u535A\u63A8zz'), [u'ah', u'\u535A', u'\u63A8', u'zz']
        )

    def test_basic_tokenizer_lower(self):
        tokenizer = bert.BasicTokenizer(do_lower_case=True)

        self.assertEqual(
            tokenizer.tokenize(u' \tHeLLo!how  \n Are yoU?  '),
            ['hello', '!', 'how', 'are', 'you', '?']
        )
        self.assertEqual(tokenizer.tokenize(u'H\u00E9llo'), ['hello'])

    def test_basic_tokenizer_no_lower(self):
        tokenizer = bert.BasicTokenizer(do_lower_case=False)

        self.assertEqual(
            tokenizer.tokenize(u' \tHeLLo!how  \n Are yoU?  '),
            ['HeLLo', '!', 'how', 'Are', 'yoU', '?']
        )

    def test_wordpiece_tokenizer(self):
        vocab_tokens = [
            '[UNK]', '[CLS]', '[SEP]', 'want', '##want', '##ed', 'wa', 'un', 'runn', '##ing',
            '[PAD]', '[MASK]'
        ]

        vocab = {}
        for (i, token) in enumerate(vocab_tokens):
            vocab[token] = i
        tokenizer = bert.WordpieceTokenizer(vocab=vocab, unk_token='[UNK]')

        self.assertEqual(tokenizer.tokenize(''), [])

        self.assertEqual(
            tokenizer.tokenize('unwanted running'), ['un', '##want', '##ed', 'runn', '##ing']
        )

        self.assertEqual(tokenizer.tokenize('unwantedX running'), ['[UNK]', 'runn', '##ing'])


class TestUtils(unittest.TestCase):

    def test_is_whitespace(self):
        self.assertTrue(utils.is_whitespace(u' '))
        self.assertTrue(utils.is_whitespace(u'\t'))
        self.assertTrue(utils.is_whitespace(u'\r'))
        self.assertTrue(utils.is_whitespace(u'\n'))
        self.assertTrue(utils.is_whitespace(u'\u00A0'))

        self.assertFalse(utils.is_whitespace(u'A'))
        self.assertFalse(utils.is_whitespace(u'-'))

    def test_is_control(self):
        self.assertTrue(utils.is_control(u'\u0005'))

        self.assertFalse(utils.is_control(u'A'))
        self.assertFalse(utils.is_control(u' '))
        self.assertFalse(utils.is_control(u'\t'))
        self.assertFalse(utils.is_control(u'\r'))
        self.assertFalse(utils.is_control(u'\U0001F4A9'))

    def test_is_punctuation(self):
        self.assertTrue(utils.is_punctuation(u'-'))
        self.assertTrue(utils.is_punctuation(u'$'))
        self.assertTrue(utils.is_punctuation(u'`'))
        self.assertTrue(utils.is_punctuation(u'.'))

        self.assertFalse(utils.is_punctuation(u'A'))
        self.assertFalse(utils.is_punctuation(u' '))
