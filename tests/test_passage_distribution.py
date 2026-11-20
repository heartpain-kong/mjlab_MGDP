import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from src.dashboard.server import read_passage_distribution
from src.tasks.velocity.mdp.passage import PassageCurriculum

class PassageDistributionTests(unittest.TestCase):
  def test_counts_include_empty_rows_and_aggregate_multiple_columns(self):
    term=object.__new__(PassageCurriculum)
    term.terrain=NS(max_terrain_level=3,terrain_levels=torch.tensor([0,1,2,1,0]),
      terrain_types=torch.tensor([0,0,1,2,2]))
    term.kind_of_column=torch.tensor([0,0,1]);term.column_names=['flat','wall','unused']
    term._status_last=0.
    with tempfile.TemporaryDirectory() as tmp:
      term.status_path=Path(tmp)/'curriculum/passage_distribution.json'
      term.write_distribution(force=True)
      data=read_passage_distribution(tmp)
      self.assertEqual(data['total'],5)
      self.assertEqual([x['counts'] for x in data['terrains']],[[1,1,1],[1,1,0],[0,0,0]])
      term.terrain.terrain_levels.fill_(2);term.write_distribution()
      self.assertEqual(read_passage_distribution(tmp),data) # throttled
      term.write_distribution(force=True)
      self.assertEqual(read_passage_distribution(tmp)['terrains'][0]['counts'],[0,0,3])
      data['total']=999;term.status_path.write_text(json.dumps(data),encoding='utf-8')
      self.assertIsNone(read_passage_distribution(tmp))

if __name__=='__main__':unittest.main()
