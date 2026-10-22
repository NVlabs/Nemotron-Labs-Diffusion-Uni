"""Small YAML mixture reader for JSON, JSONL, Parquet, Lance, and local tar images."""
import io
import json
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import yaml


@dataclass
class Example:
    prompt: str
    answer: str
    images: list
    target: object = None
    identifier: str = ''
    turns: object = None


def load_image(value, base=None):
    if isinstance(value, Image.Image):
        return value.convert('RGB')
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert('RGB')
    if isinstance(value, dict):
        value = value.get('bytes') or value.get('path') or value.get('local_url')
        return load_image(value, base)
    value = str(value)
    if value.startswith('tar_local://'):
        path, offset, length = value[len('tar_local://'):].rsplit(':', 2)
        with open(path, 'rb') as stream:
            stream.seek(int(offset)); data = stream.read(int(length))
        if len(data) != int(length):
            raise IOError('Short read from tar image reference')
        return Image.open(io.BytesIO(data)).convert('RGB')
    if '::' in value:
        archive, member = value.split('::', 1)
        archive = Path(archive)
        if not archive.is_absolute() and base: archive = Path(base) / archive
        with tarfile.open(archive) as stream:
            return Image.open(io.BytesIO(stream.extractfile(member).read())).convert('RGB')
    path = Path(value)
    if not path.is_absolute() and base: path = Path(base) / path
    return Image.open(path).convert('RGB')


def _texts(messages):
    if isinstance(messages, str): messages = json.loads(messages)
    pairs=[]
    for m in messages or []:
        role=m.get('role',m.get('from',''))
        value=m.get('content',m.get('value',''))
        if isinstance(value,list):value='\n'.join(x.get('text','') for x in value if isinstance(x,dict))
        pairs.append((role,str(value)))
    prompts=[v for r,v in pairs if r in ('user','human')]
    answers=[v for r,v in pairs if r in ('assistant','gpt')]
    # One example is the final response; preserve earlier turns as textual context.
    context='\n'.join(r+': '+v for r,v in pairs[:-2])
    return (context+'\n' if context else '')+(prompts[-1] if prompts else ''), answers[-1] if answers else ''


def normalize(row, task, base):
    prompt=row.get('prompt') or row.get('instruction') or row.get('input_prompt') or row.get('question') or row.get('caption') or ''
    answer=row.get('answer') or row.get('answer_text') or ''
    if row.get('conversations') or row.get('messages'):
        prompt,answer=_texts(row.get('conversations') or row.get('messages'))
    refs=row.get('images') or row.get('local_urls') or row.get('local_tar') or row.get('image') or []
    if task=='image_edit_opd':refs=row.get('input') or row.get('input_image') or refs
    if not isinstance(refs,list):refs=[refs]
    images=[load_image(r,base) for r in refs if r is not None]
    target=None
    if task=='t2i_sft':
        target_ref=row.get('target') or row.get('local_url') or row.get('image') or (refs[0] if refs else None)
        if target_ref is None:raise ValueError('T2I sample has no target image')
        target=load_image(target_ref,base);images=[]
    if not prompt:raise ValueError('Sample has no prompt')
    turns = row.get('conversations') or row.get('messages')
    if isinstance(turns, str): turns = json.loads(turns)
    return Example(str(prompt),str(answer),images,target,str(row.get('key',row.get('record_id',''))),turns)


class Table:
    def __init__(self,path,columns=None):
        self.path=Path(path);self.columns=columns
        suffix=self.path.suffix
        if suffix=='.parquet':
            import pyarrow.parquet as pq
            self.table=pq.ParquetFile(path);self.kind='parquet';self.size=self.table.metadata.num_rows
            self.ends=[];n=0
            for i in range(self.table.num_row_groups):n+=self.table.metadata.row_group(i).num_rows;self.ends.append(n)
            self.cached_group=None;self.cached_rows=None
        elif suffix=='.lance':
            import lance
            self.table=lance.dataset(str(path));self.kind='lance';self.size=self.table.count_rows()
        else:
            with self.path.open() as stream:
                self.table=[json.loads(line) for line in stream if line.strip()] if suffix=='.jsonl' else json.load(stream)
            if isinstance(self.table,dict):self.table=self.table.get('data',self.table.get('samples',[]))
            self.kind='json';self.size=len(self.table)
        if not self.size:raise ValueError('Empty dataset: '+str(path))

    def row(self,index):
        if self.kind=='lance':return self.table.take([index],columns=self.columns).to_pylist()[0]
        if self.kind=='json':return self.table[index]
        import bisect
        group=bisect.bisect_right(self.ends,index)
        if self.cached_group!=group:
            self.cached_rows=self.table.read_row_group(group,columns=self.columns);self.cached_group=group
        offset=index-(self.ends[group-1] if group else 0)
        return self.cached_rows.slice(offset,1).to_pylist()[0]


class Mixture:
    def __init__(self,path,task,seed=42):
        self.task=task;self.rng=random.Random(seed);self.leaves=[];self.tables={}
        self._read(Path(path).resolve(),1.,{})
        if not self.leaves:raise ValueError('No positive-weight dataset leaves')

    def _read(self,path,weight,inherited):
        spec=yaml.safe_load(path.read_text())
        for leaf in spec['datasets']:
            group=leaf.get('length_group',leaf.get('name','default'))
            factor=float(spec.get('weights',{}).get(group,1.))*float(leaf.get('sampling_factor',1.))
            if weight*factor<=0:continue
            merged={**inherited,**leaf}
            target=Path(leaf.get('yaml_path') or leaf.get('json_path') or leaf.get('path'))
            if not target.is_absolute():target=path.parent/target
            if 'yaml_path' in leaf:self._read(target.resolve(),weight*factor,merged)
            else:self.leaves.append((target.resolve(),weight*factor,merged))

    def sample(self):
        path,_,leaf=self.rng.choices(self.leaves,weights=[x[1] for x in self.leaves],k=1)[0]
        if path not in self.tables:self.tables[path]=Table(path,leaf.get('columns'))
        table=self.tables[path];index=self.rng.randrange(table.size)
        row=table.row(index)
        return normalize(row,self.task,path.parent)
