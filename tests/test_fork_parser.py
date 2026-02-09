"""Tests for fork_parser.py - parser state transitions and fork validity."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fork_parser import ForkParser, ParserState


class TestParserStateTransitions:
    def test_initial_state(self):
        p = ForkParser()
        assert p.state == ParserState.BEFORE_THINK
        assert p.fork_count == 0
        assert not p.invalid_fork
        assert not p.has_complete_answer

    def test_think_open(self):
        p = ForkParser()
        p.feed("<think>")
        assert p.state == ParserState.IN_THINK

    def test_think_open_partial(self):
        p = ForkParser()
        p.feed("<thi")
        assert p.state == ParserState.BEFORE_THINK
        p.feed("nk>")
        assert p.state == ParserState.IN_THINK

    def test_think_close(self):
        p = ForkParser()
        p.feed("<think>some reasoning</think>")
        assert p.state == ParserState.AFTER_THINK

    def test_think_close_partial(self):
        p = ForkParser()
        p.feed("<think>reasoning</thi")
        assert p.state == ParserState.IN_THINK
        p.feed("nk>")
        assert p.state == ParserState.AFTER_THINK

    def test_answer_detection(self):
        p = ForkParser()
        p.feed("<think>reasoning</think>\n<answer>42</answer>")
        assert p.has_complete_answer
        assert p.answer_text == "42"

    def test_answer_not_complete(self):
        p = ForkParser()
        p.feed("<think>reasoning</think>\n<answer>42")
        assert not p.has_complete_answer

    def test_answer_with_spaces(self):
        p = ForkParser()
        p.feed("<think>r</think>\n<answer> (1 + 2) * 3 </answer>")
        assert p.has_complete_answer
        assert p.answer_text == "(1 + 2) * 3"

    def test_no_think_tag(self):
        p = ForkParser()
        p.feed("just some text without tags")
        assert p.state == ParserState.BEFORE_THINK


class TestForkValidity:
    def test_fork_valid_in_think(self):
        p = ForkParser()
        p.feed("<think>reasoning so far")
        assert p.check_fork_valid()

    def test_fork_invalid_before_think(self):
        p = ForkParser()
        assert not p.check_fork_valid()

    def test_fork_invalid_after_think(self):
        p = ForkParser()
        p.feed("<think>reasoning</think>")
        assert not p.check_fork_valid()

    def test_fork_invalid_second_fork(self):
        p = ForkParser()
        p.feed("<think>reasoning")
        assert p.register_fork()  # first fork is valid
        assert not p.check_fork_valid()  # second would be invalid

    def test_register_valid_fork(self):
        p = ForkParser()
        p.feed("<think>reasoning")
        assert p.register_fork()
        assert p.fork_count == 1
        assert not p.invalid_fork

    def test_register_invalid_fork_before_think(self):
        p = ForkParser()
        assert not p.register_fork()
        assert p.fork_count == 1
        assert p.invalid_fork

    def test_register_invalid_fork_after_think(self):
        p = ForkParser()
        p.feed("<think>reasoning</think>")
        assert not p.register_fork()
        assert p.invalid_fork

    def test_register_two_forks(self):
        p = ForkParser()
        p.feed("<think>reasoning")
        assert p.register_fork()
        assert not p.register_fork()
        assert p.fork_count == 2
        assert p.invalid_fork


class TestParserClone:
    def test_clone_preserves_state(self):
        p = ForkParser()
        p.feed("<think>reasoning")
        p.register_fork()
        c = p.clone()
        assert c.state == p.state
        assert c.fork_count == p.fork_count
        assert c.text == p.text
        assert not c.invalid_fork

    def test_clone_independent(self):
        p = ForkParser()
        p.feed("<think>reasoning")
        c = p.clone()
        c.feed(" more text")
        assert "more text" in c.text
        assert "more text" not in p.text


class TestGetResult:
    def test_result_after_full_sequence(self):
        p = ForkParser()
        p.feed("<think>reasoning")
        p.register_fork()
        p.feed("</think>\n<answer>42</answer>")
        r = p.get_result()
        assert r.state == ParserState.AFTER_THINK
        assert r.fork_count == 1
        assert not r.invalid_fork
        assert r.has_complete_answer
        assert r.answer_text == "42"
        assert r.fork_position is not None


class TestReset:
    def test_reset_clears_state(self):
        p = ForkParser()
        p.feed("<think>reasoning")
        p.register_fork()
        p.reset()
        assert p.state == ParserState.BEFORE_THINK
        assert p.fork_count == 0
        assert not p.invalid_fork
        assert not p.has_complete_answer
