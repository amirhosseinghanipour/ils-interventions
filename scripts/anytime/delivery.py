"""Exercise the live deadline contract with a consumer in a separate CPU process."""
import argparse
import json
import os
import selectors
import subprocess
import sys
from pathlib import Path
from time import perf_counter


def clock():
    return perf_counter()*1000


def observe():
    # There is no model or GPU dependency in the independent consumer.
    print('ready', flush=True)
    spec = json.loads(sys.stdin.buffer.readline())
    end = spec['start'] + spec['cap']
    latest = dict(i=0, paid_effect=0., stage='no-op', received_ms=0.)
    buf = b''
    with selectors.DefaultSelector() as sel:
        sel.register(sys.stdin, selectors.EVENT_READ)
        while clock() < end:
            if not sel.select(max(0., (end-clock())/1000)):
                break
            packet = os.read(sys.stdin.fileno(), 65536)
            received = clock()
            if received > end or not packet:
                break
            lines = (buf + packet).split(b'\n')
            buf = lines.pop()
            for line in lines:
                event = json.loads(line)
                latest = dict(**event, received_ms=received-spec['start'])
    print(json.dumps(dict(incumbent=latest, released_ms=clock()-spec['start'])),flush=True)


def run():
    import numpy as np
    from . import benchmark, engine
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('task')
    p.add_argument('out',type=Path)
    args=p.parse_args()
    model,tok=engine.load(args.checkpoint)
    row=engine.rows(args.task,'development')[-1]
    for k in (32,64):
        prof=json.loads((args.out/str(k)/'profile.json').read_text())
        pool,_=benchmark.masks_for(k,28131)
        oracle=engine.prepare(model,tok,row,k)
        oracle.gradients(np.full((1,k),.5,np.float32))
        for arm in ['complement','hvp4']:
            child=subprocess.Popen([sys.executable, __file__, '--observer'],
                                   stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            assert child.stdout.readline().strip()=='ready'
            start=clock()
            def send(message):
                child.stdin.write(json.dumps(message)+'\n')
                child.stdin.flush()
            send(dict(start=start,cap=prof['budget_ms']))
            for event,_ in benchmark.search_stream(model,tok,row,pool,arm,8353,prof,start):
                try:
                    send({key:event[key] for key in ('i','paid_effect','stage')})
                except BrokenPipeError:
                    break
            worker_ms=clock()-start
            # Consumer keeps its own timer even if the worker ended early.
            received=json.loads(child.stdout.readline())
            try:
                child.stdin.close()
            except BrokenPipeError:
                pass
            child.stdout.close()
            child.wait()
            assert child.returncode==0
            assert received['incumbent']['received_ms'] <= prof['budget_ms']
            oracle_score=float(oracle.exact(pool[[received['incumbent']['i']]])[0])
            received.update(cap_ms=prof['budget_ms'],worker_ms=worker_ms,row=row['id'],
                            replay_effect=oracle_score,method=arm)
            benchmark.write(args.out/str(k)/f'delivery_{arm}.json',received)

        # Secondary exact-only policy. Short batches ensure progress is sent several
        # times per deadline, at some cost in steady-state device throughput.
        runs = []
        for ix, row_h in enumerate(engine.rows(args.task)):
            pool, _ = benchmark.masks_for(k, 38756+ix)
            start = clock()
            oracle_h = engine.prepare(model,tok,row_h,k)
            bound = prof['budget_ms']/6
            sizes = {int(n):t for n,t in prof['batch_ms'].items()}
            cursor, incumbent, paid = 1, 0, 0.
            events = []
            while cursor < len(pool):
                room = prof['budget_ms'] - (clock()-start)
                feasible = [n for n,t in sizes.items() if t <= min(bound, room/1.25)]
                if not feasible:
                    break
                batch = min(feasible, key=lambda n:sizes[n]/n)
                x = oracle_h.exact(pool[cursor:cursor+batch])
                j = int(np.argmax(x))
                if x[j] > paid:
                    incumbent, paid = cursor+j, float(x[j])
                cursor += len(x)
                events.append(dict(ms=clock()-start,i=incumbent,paid_effect=paid,exact=cursor-1))
            runs.append(dict(id=row_h['id'],events=events,return_ms=clock()-start))
        benchmark.write(args.out/str(k)/'streamed_exact.json',runs)


if __name__=='__main__':
    if sys.argv[1:]==['--observer']:
        observe()
    else:
        run()
