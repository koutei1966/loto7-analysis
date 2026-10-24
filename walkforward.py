import csv, json, random, math
from pathlib import Path
from collections import Counter

OUT = Path('results'); OUT.mkdir(exist_ok=True)
FEATURES = ('frequency','recent','carry','shift','gap','pair')
STRATEGIES = (
    (1.4,1.5,.24,.15,.15,.16),
    (.8,2.3,.4,.2,.10,.25),
    (2.0,.9,.2,.2,.30,.08),
)

def load():
    with open('draws.csv', encoding='utf-8-sig', newline='') as f:
        rd=csv.DictReader(f)
        if not all(f'n{i}' in (rd.fieldnames or []) for i in range(1,8)):
            raise ValueError('Missing n1..n7')
        data=[tuple(sorted(int(r[f'n{i}']) for i in range(1,8))) for r in rd]
    if len(data)<21 or any(len(set(d))!=7 or d[0]<1 or d[-1]>37 for d in data):
        raise ValueError('Invalid draw history')
    return data

def properties(c,prev):
    odds=sum(n%2 for n in c)
    consecutive=sum(b==a+1 for a,b in zip(c,c[1:]))
    triple=int(any(c[i+2]==c[i]+2 for i in range(5)))
    bands=tuple(sum(a<=n<=b for n in c) for a,b in ((1,12),(13,24),(25,37)))
    return (odds, sum(c)//15, consecutive, triple, len(set(c)&set(prev)), max(bands))

def filters(history):
    counts=[Counter() for _ in range(6)]
    for i in range(1,len(history)):
        for k,v in enumerate(properties(history[i],history[i-1])):
            counts[k][v]+=1
    minimum=math.ceil(.05*(len(history)-1))
    return [set(v for v,n in counter.items() if n>=minimum) for counter in counts]

def allowed(c,prev,keep):
    return all(v in keep[k] for k,v in enumerate(properties(c,prev)))

def context(history):
    n=len(history); last=set(history[-1]); freq=Counter(x for d in history for x in d)
    recent=Counter(x for d in history[-30:] for x in d)
    pair=Counter()
    for d in history[-120:]:
        for i,a in enumerate(d):
            for b in d[i+1:]: pair[a,b]+=1
    gap={x:n for x in range(1,38)}
    for i,d in enumerate(reversed(history)):
        for x in d:
            if gap[x]==n: gap[x]=i
    return freq,recent,pair,gap,last

def score(c,ctx,weights,history_len):
    freq,recent,pair,gap,last=ctx
    f,r,carry,shift,age,co=weights
    val=0
    for x in c:
        distance=min(abs(x-y) for y in last)
        val+=(f*freq[x]/history_len+r*recent[x]/min(30,history_len)
              +carry*(x in last)+shift*(distance in (1,2))
              +age*min(gap[x],15)/15)
    val+=co*sum(pair[a,b] for i,a in enumerate(c) for b in c[i+1:])/min(120,history_len)
    return val

def propose(history, weights, salt):
    ctx=context(history); keep=filters(history); rng=random.Random(99173+len(history)*11+salt)
    freq,recent,pair,gap,last=ctx
    # Deterministic stochastic search; no information from the target draw.
    candidates=set()
    for _ in range(460):
        c=tuple(sorted(rng.sample(range(1,38),7)))
        if allowed(c,last,keep): candidates.add(c)
    # Also search around highly ranked individual numbers.
    individual=sorted(range(1,38),key=lambda x:-(weights[0]*freq[x]/len(history)+weights[1]*recent[x]/min(30,len(history))+weights[2]*(x in last)))
    for _ in range(340):
        pool=individual[:rng.choice((14,20,28,37))]
        c=tuple(sorted(rng.sample(pool,7)))
        if allowed(c,last,keep): candidates.add(c)
    if not candidates:
        raise ValueError('5-percent filters eliminated all sampled candidates')
    return max(candidates,key=lambda c:(score(c,ctx,weights,len(history)),-sum(c)))

def three(history, performance):
    # Historical-only strategy ranking; each strategy is independently evaluated.
    order=sorted(range(3),key=lambda j:(-sum(row[j] for row in performance[-60:])/max(1,len(performance[-60:])),j))
    picks=[propose(history,STRATEGIES[j],j) for j in range(3)]
    return picks,order

def main():
    draws=load(); rows=[]; performance=[]
    for i in range(20,len(draws)):
        picks,order=three(draws[:i],performance)
        hits=[len(set(p)&set(draws[i])) for p in picks]
        performance.append(hits)
        rows.append([i+1,*['-'.join(map(str,p)) for p in picks],*hits,max(hits),order[0]+1])
    with (OUT/'walkforward_3tickets.csv').open('w',encoding='utf-8',newline='') as f:
        wr=csv.writer(f);wr.writerow(['draw_index','ticket_1','ticket_2','ticket_3','hits_1','hits_2','hits_3','best_hits','historically_best_strategy']);wr.writerows(rows)
    future,order=three(draws,performance)
    summary={'model':'adaptive_5percent_v2','draws_loaded':len(draws),'tests':len(rows),
      'average_best_hits':round(sum(r[-2] for r in rows)/len(rows),4),
      'draw_700_validation':next((r for r in rows if r[0]==700),None),
      'next_draw':len(draws)+1,'next_three_tickets':[list(p) for p in future],
      'best_strategy_from_prior_60':order[0]+1,
      'filters':'Each of six historical categorical patterns must have >=5% observed frequency, calculated using prior draws only.',
      'limitations':'Date/month effects and high-order co-occurrence not yet included; candidate search is sampled, not exhaustive.'}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__': main()
