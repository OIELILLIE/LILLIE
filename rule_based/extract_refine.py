import requests,urllib
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

objrules = [({"dobj"},(1,0,2),0),
            ({"xcomp","ccomp"},(1,0,2),2),
            ({"cop"},(1,2,0),1),
            ({"nmod"},(1,0,2),3),
            ({"nmod:tmod","nmod:npmod"},(1,0,2),4)]

subjdeps = {"nsubj","nsubjpass","csubj","csubjpass"}
objdeps = {"dobj","xcomp","ccomp","nmod","nmod:tmod","nmod:npmod"}
clausedeps = {"advcl","acl","acl:relcl","parataxis","root"}
compdeps = {"parataxis","root"}
strictobjs = {"dobj","xcomp","ccomp"}

branchrules = {"amod":((1,1,1),(0,1,2)),
               "nmod":((1,0,1),(0,2,2)),
               "nmod:tmod":((1,0,1),(0,2,2)),
               "nmod:npmod":((1,0,1),(0,2,2)),
               "vmod":((0,1,0),(-1,1,1)),
               "cop":((0,1,1),(-1,1,1)),
               "det":((1,1,1),(0,1,2)),
               "punct":((0,0,0),(0,0,0)),
               "case":((1,1,1),(0,1,1)),
               "nummod":((1,1,1),(0,1,2)),
               "mark":((0,0,0),(-1,-1,-1)),
               "tomark":((0,1,0),(-1,1,1)),
               "mwe":((1,0,1),(0,-1,2)),
               "cc":((1,0,1),(0,-1,2)),
               "cc:preconj":((1,0,1),(0,-1,2)),
               "det:predet":((1,0,1),(0,-1,2)),
               "nmod:poss":((1,0,1),(0,-1,2)),
               "compound":((1,1,1),(0,1,2)),
               "compound:prt":((0,1,1),(-1,1,1)),
               "conj":((1,0,1),(0,-1,2)),
               "aux":((0,1,1),(-1,1,1)),
               "auxpass":((0,1,0),(-1,1,1)),
               "appos":((1,0,1),(0,-1,2)),
               "neg":((1,1,1),(0,1,2)),
               "nsubj":((0,0,1),(-1,-1,2)),
               "advcl":((0,0,1),(-1,-1,2)),
               "acl":((0,0,1),(0,-1,2)),
               "acl:relcl":((0,0,1),(0,-1,2)),
               "advmod":((0,1,1),(0,1,1)),
               "dobj":((0,0,1),(-1,-1,2)),
               "xcomp":((0,0,1),(-1,2,2)),
               "parataxis":((0,0,0),(-1,-1,-1)),
               "nsubjpass":((0,0,1),(-1,-1,2)),
               "ccomp":((0,0,1),(-1,-1,2)),
               "dep":((0,0,1),(-1,2,2)),
               "root":((0,0,1),(-1,-1,2)),
               "expl":((0,0,0),(-1,-1,-1)),
               "csubj":((0,0,0),(-1,-1,-1)),
               "iobj":((0,1,0),(-1,1,1)),
               "block":((0,0,0),(-1,-1,-1)),
               "whsubj":((0,0,1),(-1,-1,2)),
               "prpsubj":((0,0,1),(-1,2,2)),
               "csubjpass":((0,0,1),(-1,-1,2)),
               "discourse":((0,0,0),(-1,-1,-1)),
              }

def triples(prs,**kwargs):
    res = {(*(expterms(svot,i) for i,_ in enumerate(svot)),)
           for t in treeforms(prs,**kwargs) for svot in basetoks(prs,t)}
    return [[" ".join([prs["tokens"][w] for w in sorted(ws)]) for ws in svo] for svo in res]

def basetoks(prs,t):
    svos = []
    for (p,_,pt),(c1,d1,b1),(c2,d2,b2) in treetrips(t):
        if d1 not in subjdeps: continue
        for ds,sw,rd in objrules:
            if d2 in ds:
                pt = [(c,*db) for c,*db in pt if c not in (c1,c2)]
                svos.append((rd,[(p,c1,c2)[i] for i in sw],
                             *(((p,c1,c2)[i],"base",(pt,b1,b2)[i]) for i in sw)))
    return uniquesvos(prs,svos)

def uniquesvos(prs,svos):
    return {(sn,vn):((sn,*s),(vn,*v),o) for _,_,(sn,*s),(vn,*v),o in reversed(sorted(svos))
            if posmatch(prs,(vn,),"V")}.values()
    
def tconv(prs):
    return [prs["root"],"ROOT",rtconv(prs["deps"],prs["root"])]

def rtconv(g,r):
    n = []
    for d in g[r]:
        for nn in g[r][d]:
            n.append([nn,d,rtconv(g,nn)])
    return n

def coref(prs,t,clausal):
    ts = []
    subjmark(prs,t)
    for p,nt in clausetrees(t):
        for i,_ in enumerate(nt[2]):
            subjswap(p,nt[2],i)
        ts.append(nt)
    ts.append(tcopy(t))
    return ts

def subjswap(p,nt,i):
    if nt[i][1] == "whsubj":
        nt[i] = [p[0],"nsubj",chfilt(p,clausedeps|{"case"})]
    elif nt[i][1] == "prpsubj":
        for c in p[2]:
            if c[1] in subjdeps:
                nt[i] = tcopy(c)
                break
        else:
            nt[i][1] = "nsubj"

def clausetrees(t):
    return {c[0]:(p,tcopy(c)) for p,c in pcprs(t) if c[1] in clausedeps}.values()

def vsplit(prs,t):
    for v,s,c in vscs(t):
        if posmatch(prs,v,"V") and c[1] == "conj":
            if not hastrips(c):
                if s[1] in subjdeps:
                    c[2].append(tcopy(s))
            else:
                c[1] = "block"
                
def vscs(t):
    for v in bfs(t):
        for s in v[2]:
            for c in v[2]:
                yield v,s,c
    
def remclauses(t):
    for b in dfs(t):
        if b[1] in clausedeps:
            b[1] = "block"
                        
def subjmark(prs,t):
    for p,c in pcprs(t):
        if p[1] in clausedeps:
            bmod(prs,c,"W",subjdeps,"whsubj")
        if p[1] in compdeps:
            bmod(prs,c,"PRP",subjdeps,"prpsubj")
        bmod(prs,p,"T","mark","tomark")
                
def pcprs(t):
    for b in bfs(t):
        for c in b[2]:
            yield (b,c)
        
def bmod(prs,b,pos,ds,ch):
    if posmatch(prs,b,pos) and b[1] in ds:
        b[1] = ch
        
def posmatch(prs,n,pos):
    return prs["pos"][n[0]][:len(pos)] == pos

def treeforms(prs,enhanced_predicates=True,nominal_conjunctions=False,
              expanded_context=False,verbal_conjunctions=True):
    t = tconv(prs)
    if verbal_conjunctions:
        vsplit(prs,t)
    ts = coref(prs,t,expanded_context)
    if nominal_conjunctions:
        ts = [nt for t in ts for nt in nconjsplit(prs,t)]
    for t in ts:
        if not expanded_context:
            remclauses(t)
        if enhanced_predicates:
            verbredis(prs,t)
    return ts

def nconjsplit(prs,t):
    nts = [remconjs(t)]
    for p,c in pcprs(t):
        if not posmatch(prs,p,"V"):
            if c[1] == "conj" and posmatch(prs,c,"N"):
                p[0],p[2] = c[0],chfilt(p,{"compound","conj","cc"})+chfilt(c,{})
                nts.append(tcopy(t))
    return nts

def remconjs(t):
    nt = tcopy(t)
    for b in dfs(nt):
        b[2] = chfilt(b,{"conj","cc"})
    return nt

def chfilt(b,ds):
    return [tcopy(sb) for sb in b[2] if sb[1] not in ds]
        
def verbredis(prs,t):
    for p,c in pcprs(t):
        if not any([s[1] in objdeps for s in p[2]]):
            continue
        copexchange(p,c)
        if posmatch(prs,p,"V") and stricttrips(p):
            nmodswitch(p)
                        
def copexchange(a,c):
    if c[1] == "cop":
        a[0],c[0],c[1] = c[0],a[0],"advmod"
        
def nmodswitch(t):
    for n in t[2]:
        if n[1][:4] == "nmod":
            n[1] = "vmod"
        
def stricttrips(t):
    return (any([n[1] in subjdeps for n in t[2]]) and
            any([n[1] in strictobjs for n in t[2]]))
    
def tcopy(t):
    return [*t[:2],[tcopy(st) for st in t[2]]]
    
def hastrips(t):
    return any([d1 in subjdeps and any([d2 in ds for ds,_,_ in objrules])
                for _,(_,d1,_),(_,d2,_) in treetrips(t)])

def treetrips(t):
    _,_,st = t
    for b1 in st:
        for b2 in st:
            yield (t,b1,b2)
        yield from treetrips(b1)
        
def dfs(t):
    for b in t[2]:
        yield from dfs(b)
    yield t
    
def bfs(t):
    q = [t]
    while q:
        c,q = q[0],q[1:]
        yield c
        q.extend(c[2])
    
def expterms(svot,i):
    return branchfilt([*svot[i][:2],branchcoll(svot,i)],i)
    
def branchcoll(svot,i):
    return [[n,d,b] for j,(r,_,st) in enumerate(svot)
            for n,d,b in st if branchrules[d][1][j] == i]

def branchfilt(b,i):
    (n,d,t),r = b,()
    for nn,dd,st in t:
        if branchrules[dd][0][i]:
            r += (branchfilt([nn,dd,st],i))
    return (n,) + r


url = "http://localhost:10000/?properties={%s}" % urllib.parse.quote_plus('"annotators":"depparse","outputFormat":"json"')

def parseconv(r):
    snt,dps = r.json()["sentences"][0],[]
    for i,_ in enumerate(snt["tokens"]):
        dps.append({})
        for d in snt["basicDependencies"]:
            if d["governor"] == i + 1:
                dps[i][d["dep"]] = dps[i].get(d["dep"],[]) + [d["dependent"]-1]
    return {"tokens":[td["word"] for td in snt["tokens"]],
            "pos":[td["pos"] for td in snt["tokens"]],
            "root":[d["dependent"]-1 for d in snt["basicDependencies"] if d["dep"] == "ROOT"][0],
            "deps":dps}

def parsetxt(txt):
    r = requests.post(url,data=txt.encode("ascii",errors="ignore").decode())
    if r.status_code == 200 and txt:
        return parseconv(r)
    else:
        raise Exception("Parse error")

parsetxt("This is a test.")

swset = set(stopwords.words("english"))

def combtrips(prs,zhtrips,inftrips,infotrips):
    ctrips,dn,posmap = [],set(),dict(zip(prs["tokens"],prs["pos"]))
    for trip in zhtrips + sorted(inftrips,key=lambda x:-len(x[2].split())):
        s = tuple(t for t in trip[1].split() if t not in swset and t in posmap and posmap[t][:1] == "V")
        if all([posmap[t] not in {"VBD","VB","VBN","VBP","VBZ"}
                for t in trip[1].split() if t in prs["tokens"]]):
            continue
        if s not in dn:
            dn.add(s)
            ctrips.append(trip)
    return ctrips+infotrips

def triprefine(snt,trips,keep=False):
    ntrips,otrips = [],[]
    for trip in trips:
        antrips = treeprune(snt,trip)
        for ntrip in antrips:
            if all(map(bool,ntrip)):
                ntrips.append(ntrip)
        if not antrips and keep:
            otrips.append(list(trip))
    return (list({tuple([" ".join([snt["tokens"][w] for w in sorted(ws)])
                         for ws in svo]) for svo in ntrips}),otrips)

def toknums(ws,snt):
    ns,c,ts = [-1 for _ in ws],0,list(snt["tokens"])
    for i,_ in enumerate(ws):
        for j,_ in enumerate(ts[:-(i-c)-1]):
            if ws[c:i+1] == ts[j:j+(i-c)+1]:
                ns[c:i+1] = range(j,j+(i-c)+1)
                break
        else:
            c = i
            for j,t in enumerate(ts):
                if t == ws[i]:
                    ns[i] = j
                    break
    return ns

def treeprune(prs,trip):
    s,v,o,res = *[{t for t in toknums(ws,prs) if t >= 0} for ws in (ws.split() for ws in trip)],[]
    ts = treeforms(prs,enhanced_predicates=False,nominal_conjunctions=False,
                   expanded_context=False,verbal_conjunctions=False)
    for t in ts:
        nsvos = [s,v,o]
        for i,ws in enumerate(nsvos):
            cmps = sorted([cmp for cmp in {tuple(conncomp(t,ws,w)) for w in ws}],
                          key=len,reverse=True)
            if not cmps:
                nsvos[i] = tuple()
            else:
                for n in bfs(t):
                    if n[0] in cmps[0]:
                        break
                nsvos[i] = tcopy((n[0],"base",n[2]))
        if not all(map(bool,nsvos)):
            continue
        nsvos = [[n[0],n[1],[b for b in n[2] if b[0] not in [nn[0] for nn in nsvos]]]
                 for i,n in enumerate(nsvos)]
        ns,nv,no = [set(expterms(nsvos,i)) for i,_ in enumerate(nsvos)]
        res.append((ns-no,nv-ns-no,no))
    return res

def conncomp(t,ws,w):
    cmp = set()
    for n in bfs(t):
        if n[0] == w:
            for c in bfs(n):
                if c[0] in ws:
                    cmp.add(c[0])
            break
    return cmp

def combinetriples(snt,trips,**kwargs):
    prs = parsetxt(snt)
    return combtrips(prs,triples(prs,**kwargs),*triprefine(prs,trips,keep=True))
    
    
if __name__ == "__main__":
    import argparse,csv
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', dest='infile', required=True,
                    metavar='INPUT_FILE', help='The input file to the script.')
    inp = parser.parse_args().infile
    outp,sents = [],{}
    with open(inp,"r") as fl:
        for ln in csv.DictReader(fl):
            sents[ln["sent_rawtext"]] = (sents.get(ln["sent_rawtext"]) +
                                         [(ln["subject"],ln["predicate"],ln["object"])])
    outp = [trip for sent in sents
            for trip in combinetriples(sent,sents["sent"])]
    with open("./output.txt","w") as fl:
        fl.write("\n".join([" ; ".join(trip) for trip in outp]))

