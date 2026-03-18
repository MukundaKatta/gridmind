"""Tests for Gridmind."""
from src.core import Gridmind
def test_init(): assert Gridmind().get_stats()["ops"] == 0
def test_op(): c = Gridmind(); c.generate(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Gridmind(); [c.generate() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Gridmind(); c.generate(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Gridmind(); r = c.generate(); assert r["service"] == "gridmind"
