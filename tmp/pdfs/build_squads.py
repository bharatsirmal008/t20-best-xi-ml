import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('D:/ADS')
TEAMS = {129:'Chennai Super Kings',252:'Delhi Capitals',615:'Gujarat Titans',6:'Kolkata Knight Riders',614:'Lucknow Super Giants',3:'Mumbai Indians',494:'Punjab Kings',134:'Rajasthan Royals',1:'Royal Challengers Bengaluru',2:'Sunrisers Hyderabad'}

def select():
    import pandas as pd
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    tree = ast.parse((ROOT/'streamlit_app/app.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'optimize_xi')
    env = dict(pd=pd, np=np, milp=milp, LinearConstraint=LinearConstraint, Bounds=Bounds)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<optimizer>', 'exec'), env)
    p = pd.read_csv(ROOT/'files/players_performance_final.csv')
    t = pd.read_csv(ROOT/'results/phase7_historical_player_match_training.csv', usecols=['team_id','player_id'])
    countries = pd.read_csv(ROOT/'results/phase9_country_catalog.csv')['country'].sort_values()
    pools = [('IPL',name,p[p.ID.isin(t.loc[t.team_id.eq(tid),'player_id'])]) for tid,name in TEAMS.items()]
    pools += [('Country',name,p[p.country.eq(name)]) for name in countries]
    out=[]
    for mode,name,pool in pools:
        pool = pool.copy()
        norm = pool.name.fillna('').str.lower().str.replace(r'[^a-z0-9]+',' ',regex=True).str.strip()
        pool = pool[~norm.duplicated(keep=False) & norm.ne('')].copy()
        pool['suitability_score'] = pd.to_numeric(pool.total_T20_matches,errors='coerce').fillna(0) + .001
        pool = pool.sort_values(['suitability_score','name'],ascending=[False,True])
        feasible=False
        try:
            xi,_=env['optimize_xi'](pool,mode.upper())
            chosen=pd.concat([pool[pool.ID.isin(xi.ID)],pool[~pool.ID.isin(xi.ID)].head(4)])
            feasible=True
        except ValueError:
            chosen=pool.head(15)
        chosen=chosen.sort_values('name')
        assert chosen.name.nunique()==len(chosen)
        assert len(chosen)<=15
        out.append(dict(mode=mode,name=name,available=len(pool),feasible=feasible,players=chosen[['name','playing_role_clean']].to_dict('records')))
    print(json.dumps(out))

def build():
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from pypdf import PdfReader
    records=json.loads(subprocess.check_output([str(ROOT/'.venv_api/Scripts/python.exe'),__file__,'--select'],text=True))
    output=ROOT/'IPL_and_Country_15_Player_Test_Squads.pdf'
    c=canvas.Canvas(str(output),pagesize=A4)
    c.setTitle('IPL and Country - Test Squad Directory')
    w,h=A4
    def text(x,y,s,size=11,bold=False,color='#243b30'):
        c.setFillColor(HexColor(color)); c.setFont('Helvetica-Bold' if bold else 'Helvetica',size); c.drawString(x,y,s)
    def page_header(section,title):
        c.setFillColor(HexColor('#123d2e')); c.rect(0,h-130,w,130,fill=1,stroke=0)
        text(44,h-39,'SELECTION ROOM  /  TEST SQUAD DIRECTORY',9,True,'#c8e591')
        text(44,h-77,title,23,True,'#ffffff')
        text(44,h-103,section,10,False,'#d0dfd5')
    def footer():
        text(44,33,'Local catalog test data | Historical membership | 22 September 2026',8,False,'#66776d')
        text(w-72,33,str(c.getPageNumber()),9)
        c.showPage()
    page_header('10 IPL teams + 43 countries','15-player test squads')
    lines=[
        'How to use this directory',
        '1. Select IPL / Franchise or Country in the app.',
        '2. Select the exact team or country named on the squad page.',
        '3. For IPL, select a different opponent team.',
        '4. Choose Enter numbered names and paste the player-name list.',
        '5. Check the matched count, set the venue, and generate Best XI.',
        '',
        'What these squads represent',
        'Names come directly from the local player and membership files.',
        'These are historical test pools, not official current-season squads.',
        'Players were prioritized by recorded T20 experience, with a feasible',
        '11-player role composition selected first where the data permits.',
        'Feasibility uses the app optimizer, including the IPL overseas cap.',
        'This checks squad constraints, not every possible match prediction.',
        '',
        'Limited country pools',
        'Countries with fewer than 15 unique eligible names list all available.',
        'A warning identifies squads that cannot produce a valid XI.',
        'Do not mix names from another team or country to fill a short squad.',
        '',
        'Sources in D:/ADS',
        'files/players_performance_final.csv',
        'results/phase7_historical_player_match_training.csv',
        'results/phase9_country_catalog.csv',
        'streamlit_app/app.py: team options and optimizer rules',
    ]
    for i,line in enumerate(lines): text(44,h-165-i*21,line,10.5,i in [0,7,15,20])
    footer()
    for r in records:
        page_header(r['mode']+' mode / exact eligible catalog names',r['name'])
        n=len(r['players'])
        text(44,h-162,f'{n} player names | {r["available"]} unique eligible names in catalog',11,True)
        if r['feasible']:
            status='Validated: this squad contains a feasible XI under the app rules.'
        else:
            status='Limited pool: no feasible XI under the app role / size rules.'
        text(44,h-184,status,10,False,'#176b48' if r['feasible'] else '#994718')
        if n<15: text(44,h-204,'Fewer than 15 names are available; the list below is not padded.',10,False,'#994718')
        text(44,h-239,'COPY THE NUMBERED NAMES BELOW INTO THE APP',9,True,'#66776d')
        for i,row in enumerate(r['players'],1):
            y=h-275-(i-1)*29
            if i%2:
                c.setFillColor(HexColor('#f0f4ee')); c.roundRect(38,y-9,w-76,27,4,fill=1,stroke=0)
            text(49,y,f'{i}. {row["name"]}',12)
        footer()
    c.save()
    pdf=PdfReader(output)
    assert len(pdf.pages)==54
    for i,r in enumerate(records,1):
        extracted=pdf.pages[i].extract_text()
        for p in r['players']: assert p['name'] in extracted,(r['name'],p['name'])
    print(f'Created {output}; {len(pdf.pages)} pages; all player names verified.')
    print('Full squads:',sum(len(r['players'])==15 for r in records))
    print('Limited pools:',[(r['name'],len(r['players'])) for r in records if len(r['players'])<15])

if __name__=='__main__':
    select() if '--select' in sys.argv else build()
