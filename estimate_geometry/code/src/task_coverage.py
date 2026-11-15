"""Explicit task coverage over PROVIDED object identities, without reference pose."""
import itertools
import numpy as np
from .collision import box_contact


def pair_status(geometry,states):
    lookup={int(s):i for i,s in enumerate(geometry['source_indices'])}
    if len(lookup)!=len(geometry['source_indices']):raise ValueError('Duplicate geometry sources')
    groups={}
    for source,row in enumerate(states):groups.setdefault((row['episode'],row['step']),[]).append(source)
    result=[]
    for (ep,step),sources in sorted(groups.items()):
        for a,b in itertools.combinations(sources,2):
            record=dict(episode=ep,step=step,sources=[a,b],tracks=[states[a]['object_key'],states[b]['object_key']])
            absent=[s for s in (a,b) if s not in lookup]
            if absent:record.update(status='unknown',reason='missing_geometry',missing_sources=absent)
            else:
                boxes=[{key:geometry[key][lookup[s]] for key in ('center','rotation','dims')} for s in (a,b)]
                contact=box_contact(*boxes)
                record.update(status='overlap' if contact['overlap'] else 'separated',**contact)
            result.append(record)
    return result


def evaluate_pair_coverage(geometry,reference,states):
    predictions=pair_status(geometry,states);summary={};details=[]
    for record in predictions:
        ep=str(record['episode'])
        totals=summary.setdefault(ep,dict(expected=0,answered=0,unknown=0,wrong_answered=0,
            reference_overlap=0,overlap_answered_correctly=0,overlap_unknown=0,false_separated=0,false_overlap=0))
        boxes=[{key:reference[key][s] for key in ('center','rotation','dims')} for s in record['sources']]
        truth=box_contact(*boxes)['overlap'];totals['expected']+=1;totals['reference_overlap']+=int(truth)
        item=dict(record,reference_overlap=truth)
        if record['status']=='unknown':
            totals['unknown']+=1;totals['overlap_unknown']+=int(truth)
        else:
            pred=record['status']=='overlap';totals['answered']+=1;totals['wrong_answered']+=int(pred!=truth)
            totals['overlap_answered_correctly']+=int(truth and pred)
            totals['false_separated']+=int(truth and not pred);totals['false_overlap']+=int(pred and not truth)
        details.append(item)
    for item in summary.values():
        item['answer_coverage']=item['answered']/item['expected'] if item['expected'] else None
        item['conditional_accuracy']=1-item['wrong_answered']/item['answered'] if item['answered'] else None
        item['correct_answer_fraction']=(item['answered']-item['wrong_answered'])/item['expected'] if item['expected'] else None
    return dict(episodes=summary,pairs=details,meaning='Provided-identity cuboid task coverage, not detection completeness or physical collision accuracy. Unknown is neither correct nor an automatic negative.')
