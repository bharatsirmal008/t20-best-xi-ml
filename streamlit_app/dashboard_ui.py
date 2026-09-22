"""Presentation helpers for the cricket selection dashboard."""
from html import escape
import hashlib

import streamlit.components.v1 as components

import streamlit as st


def apply_theme():
    st.markdown("""
    <style>
    /* Inherit Streamlit's selected theme, including manual Light/Dark changes.
       currentColor keeps surfaces and borders in the same palette as text. */
    .stApp { --surface:color-mix(in srgb,currentColor 5%,transparent); --edge:color-mix(in srgb,currentColor 20%,transparent); }
    [data-testid="stMainBlockContainer"] { max-width:1500px; padding:4rem 3rem; }
    [data-testid="stSidebar"] { border-right:1px solid var(--edge); }
    [data-testid="stSidebar"] h2 { font-size:1.05rem; letter-spacing:-.02em; }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top:.6rem; }
    h1,h2,h3 { letter-spacing:-.035em; }
    h3 { font-size:1.3rem !important; }
    [data-testid="stMetric"] { background:var(--surface); border:1px solid var(--edge); border-radius:16px; padding:20px; min-height:114px; }
    [data-testid="stMetricValue"] { font-size:1.65rem; white-space:normal; overflow-wrap:anywhere; }
    [data-testid="stButton"] button { border-radius:10px; transition:transform .18s,box-shadow .18s; }
    [data-testid="stButton"] button:hover { transform:translateY(-2px); box-shadow:0 7px 18px #163d2318; }
    [data-testid="stButton"] button[kind="primary"] { background:#176b48; border-color:#176b48; color:#fff; min-height:48px; }
    [data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="textarea"] { border-radius:10px; }
    [data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid var(--edge); }
    [data-testid="stAlert"] { border-radius:12px; }
    button:focus-visible,input:focus-visible,textarea:focus-visible { outline:3px solid #b9d865 !important; outline-offset:3px; }
    .brand { display:flex; align-items:center; gap:12px; margin:0 0 24px; }
    .brand-icon { background:#176b48; color:#d6f086; border-radius:12px; font-size:22px; padding:9px 13px; font-weight:800; }
    .brand strong { display:block; font-size:20px; letter-spacing:-.7px; }
    .brand small { color:inherit; opacity:.8; font-size:10px; letter-spacing:2px; }
    .dash-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:26px; gap:12px; }
    .eyebrow { font-size:10px; font-weight:700; letter-spacing:2px; color:inherit; text-transform:uppercase; }
    .dash-top h1 { margin:6px 0 0; font-size:32px; font-weight:750; padding:0; }
    .status { color:inherit; font-size:12px; padding:8px 13px; border:1px solid var(--edge); border-radius:30px; background:var(--surface); white-space:nowrap; }
    .status i { display:inline-block; width:7px; height:7px; background:#36835c; border-radius:50%; margin-right:7px; }
    .hero { position:relative; overflow:hidden; border-radius:23px; background:#123d2e; padding:38px; color:#fff; min-height:280px; margin-bottom:22px; animation:arrive .6s ease-out; }
    .hero-copy { position:relative; z-index:2; width:63%; }
    .hero .eyebrow { color:#c1db84; }
    .hero h2 { color:#fff; font-size:40px; line-height:1.15; margin:18px 0 14px; padding:0; max-width:550px; }
    .hero p { color:#c1d1c6; max-width:450px; font-size:14px; line-height:1.7; }
    .hero-tag { display:inline-block; border:1px solid #52705b; border-radius:20px; padding:5px 10px; font-size:11px; margin:10px 5px 0 0; color:#dce7d6; }
    .field { position:absolute; width:245px; height:245px; right:6%; top:24px; border:1px solid #64865c; border-radius:50%; background:radial-gradient(ellipse,#416b4140,transparent); animation:float 7s ease-in-out infinite; }
    .field:before { content:''; position:absolute; inset:30px; border:1px dashed #75996570; border-radius:50%; }
    .pitch { position:absolute; width:35px; height:100px; top:72px; left:105px; border:1px solid #b6ca8290; background:#becb7822; }
    .pitch:before,.pitch:after { content:''; position:absolute; left:-5px; right:-5px; border-top:1px solid #c3d68b; }
    .pitch:before { top:12px; } .pitch:after { bottom:12px; }
    .player-dot { position:absolute; width:9px; height:9px; border-radius:50%; background:#d4ed91; box-shadow:0 0 0 5px #cee78a13; }
    .section-line { display:flex; justify-content:space-between; align-items:center; margin:28px 0 14px; }
    .section-line h3 { margin:0; padding:0; } .section-line span { font-size:12px; color:inherit; opacity:.8; }
    .step-card { color:inherit; background:var(--surface); padding:24px; border:1px solid var(--edge); border-radius:16px; min-height:192px; transition:transform .2s,border-color .2s; }
    .step-card:hover { transform:translateY(-4px); border-color:#9fba84; }
    .step-number { display:inline-flex; align-items:center; justify-content:center; border-radius:10px; background:#eef3e6; color:#4a7040; width:34px; height:34px; font-size:12px; font-weight:700; }
    .step-card h4 { margin:18px 0 8px; font-size:16px; color:inherit; }
    .step-card p { font-size:13px; color:inherit; opacity:.85; line-height:1.65; margin:0; }
    .ready-panel { color:inherit; border:1px dashed var(--edge); border-radius:16px; padding:26px; text-align:center; margin-top:24px; background:var(--surface); }
    .ready-panel strong { display:block; color:inherit; font-size:17px; margin-bottom:8px; }
    .ready-panel p { margin:0; color:inherit; opacity:.85; font-size:13px; }
    .footer-note { font-size:11px; color:inherit; opacity:.8; margin-top:26px; text-align:center; }
    .roster { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:20px; }
    .roster-card { color:inherit; background:var(--surface); border:1px solid var(--edge); border-radius:14px; padding:18px; animation:arrive .45s ease-out both; }
    .roster-card b { display:block; margin:10px 0 4px; font-size:14px; }
    .roster-card small { color:inherit; opacity:.85; font-size:11px; }
    @keyframes arrive { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
    @keyframes float { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-7px); } }
    @media(prefers-reduced-motion:reduce) { *,*:before,*:after { animation:none !important; transition:none !important; } }
    @media(max-width:800px) { [data-testid="stMainBlockContainer"] { padding:4rem 1.4rem 2rem; } .hero { padding:26px; } .hero h2 { font-size:30px; } .hero-copy { width:100%; } .field { opacity:.18; right:-65px; } .dash-top h1 { font-size:25px; } .status { display:none; } }
    </style>
    """, unsafe_allow_html=True)


def header():
    st.markdown("""
    <div class="dash-top"><div><div class="eyebrow">THE SELECTION ROOM / T20 INTELLIGENCE</div><h1>Team dashboard</h1></div><div class="status"><i></i>Local model · Ready</div></div>
    <div class="hero"><div class="hero-copy"><div class="eyebrow">YOUR SQUAD. A SMARTER STARTING XI.</div><h2>Great teams begin<br>with better decisions.</h2><p>Turn your squad into a balanced playing XI with match-aware scoring and a selection you can review.</p><span class="hero-tag">ML player scoring</span><span class="hero-tag">Balanced team selection</span><span class="hero-tag">You make the final call</span></div>
    <div class="field" aria-hidden="true"><div class="pitch"></div><i class="player-dot" style="top:33px;left:115px"></i><i class="player-dot" style="top:74px;left:47px"></i><i class="player-dot" style="top:62px;left:180px"></i><i class="player-dot" style="top:127px;left:23px"></i><i class="player-dot" style="top:136px;left:209px"></i><i class="player-dot" style="top:195px;left:62px"></i><i class="player-dot" style="top:206px;left:155px"></i><i class="player-dot" style="top:93px;left:118px"></i><i class="player-dot" style="top:151px;left:118px"></i><i class="player-dot" style="top:174px;left:177px"></i><i class="player-dot" style="top:164px;left:52px"></i></div></div>
    """, unsafe_allow_html=True)


def sidebar_brand():
    st.markdown('<div class="brand"><div class="brand-icon">XI</div><div><strong>Selection room</strong><small>CRICKET INTELLIGENCE</small></div></div>', unsafe_allow_html=True)


def welcome(data):
    cols = st.columns(3)
    cols[0].metric("Players in the catalog", f'{len(data["players"]):,}')
    cols[1].metric("Franchises in the catalog", f'{data["team_catalog"]["team_id"].nunique():,}')
    cols[2].metric("Countries in the catalog", f'{data["country_catalog"]["country"].nunique():,}')
    st.markdown('<div class="section-line"><h3>Your next winning combination</h3><span>Three steps to your XI</span></div>', unsafe_allow_html=True)
    steps = [("01", "Set the match", "Choose your team and opponent, then add the venue and match context in the sidebar."), ("02", "Build your squad", "Paste names, upload a list, or add players individually. Match at least 11 eligible players."), ("03", "Make it your XI", "Generate your selection, explore player scores, and review the final lineup before confirming.")]
    for col, (number, title, body) in zip(st.columns(3), steps):
        col.markdown(f'<div class="step-card"><span class="step-number">{number}</span><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="ready-panel"><strong>Your playing XI starts here</strong><p>Use the sidebar to set up your match and squad, then select Generate Best XI.</p></div><div class="footer-note">Venue informs model scoring. Pitch and toss are recorded as context only.</div>', unsafe_allow_html=True)


def lineup_cards(xi):
    cards = []
    for index, (_, row) in enumerate(xi.iterrows(), 1):
        name = escape(str(row["name"]))
        role = escape(str(row["playing_role_clean"]))
        cards.append(f'<div class="roster-card" style="animation-delay:{index * .025}s"><span class="step-number">{index:02}</span><b>{name}</b><small>{role}</small></div>')
    st.markdown('<div class="roster">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def ground_view(xi):
    """Show every selected player once, grouped by catalog playing role."""
    groups = {"Wicketkeepers": [], "Batters": [], "All-rounders": [], "Bowlers": [], "Other roles": []}
    badges = {"Wicketkeepers": "WK", "Batters": "BAT", "All-rounders": "AR", "Bowlers": "BOWL", "Other roles": "XI"}
    for _, row in xi.iterrows():
        role = str(row.get("playing_role_clean", "")).lower()
        normalized = role.replace("-", "").replace(" ", "")
        if "keeper" in normalized:
            group = "Wicketkeepers"
        elif "allround" in normalized:
            group = "All-rounders"
        elif "bowl" in normalized:
            group = "Bowlers"
        elif "bat" in normalized:
            group = "Batters"
        else:
            group = "Other roles"
        groups[group].append(str(row["name"]))

    rows = []
    for group, names in groups.items():
        if not names:
            continue
        players = ''.join(
            f'<div class="ground-player"><span class="ground-shirt" aria-hidden="true">{badges[group]}</span>'
            f'<span class="ground-name">{escape(name)}</span></div>'
            for name in names
        )
        rows.append(f'<section class="ground-role" aria-label="{group}"><div class="ground-role-label">{group} · {len(names)}</div><div class="ground-players">{players}</div></section>')

    st.subheader("Your XI on the ground")
    st.caption("Drag a jersey or name to arrange your XI. You can also focus a player and use arrow keys. This visual layout does not change player roles or model scores.")
    layout_key = hashlib.sha256('|'.join(sorted(str(v) for v in xi['ID'])).encode()).hexdigest()[:20]
    components.html("""
    <style>
    body { margin:0; font-family:Arial,sans-serif; }
    .ground-toolbar { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:10px 14px; border-radius:10px; background:#102e24; color:#e6f1df; font-size:12px; margin-bottom:10px; }
    #reset { border:1px solid #9cb77b; border-radius:7px; background:#d9ed99; color:#153526; font-weight:700; padding:9px 12px; cursor:pointer; }
    .ground-player { cursor:grab; touch-action:none; user-select:none; box-sizing:border-box; }
    .ground-player:focus-visible { outline:3px solid #fff; outline-offset:4px; border-radius:8px; }
    .ground-player.dragging { cursor:grabbing; z-index:10; filter:drop-shadow(0 5px 6px #0008); }
    .ground-stadium { position:relative; isolation:isolate; overflow:hidden; max-width:1000px; margin:4px auto 28px; padding:36px 7%; border:8px solid #354b43; border-radius:44% 44% 38% 38% / 15% 15% 15% 15%; background:repeating-linear-gradient(0deg,#205c3e 0px,#205c3e 70px,#246645 70px,#246645 140px); box-shadow:inset 0 0 0 7px #122f24,inset 0 0 60px #0e2b2670; }
    .ground-stadium:before { content:''; position:absolute; z-index:-1; inset:22px; border:2px solid #cfebc280; border-radius:48%; pointer-events:none; }
    .ground-stadium:after { content:''; position:absolute; z-index:-1; width:60px; height:180px; top:50%; left:50%; transform:translate(-50%,-50%); background:#d3bd7840; border:2px solid #edd59c70; box-shadow:inset 0 16px #e7ce9220,inset 0 -16px #e7ce9220; }
    .ground-role { position:relative; text-align:center; margin:12px 0 26px; }
    .ground-role:last-child { margin-bottom:10px; }
    .ground-role-label { display:inline-block; padding:4px 12px; border-radius:20px; background:#102e24; color:#e6f1df; font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-bottom:16px; }
    .ground-players { display:flex; justify-content:center; flex-wrap:wrap; gap:16px 12px; }
    .ground-player { display:flex; flex-direction:column; align-items:center; gap:7px; width:130px; animation:arrive .45s ease-out both; }
    .ground-shirt { display:flex; align-items:center; justify-content:center; width:48px; height:48px; background:#d9ed99; color:#153526; font-size:11px; font-weight:800; clip-path:polygon(22% 0,38% 0,42% 9%,58% 9%,62% 0,78% 0,100% 22%,83% 39%,77% 29%,77% 100%,23% 100%,23% 29%,17% 39%,0 22%); filter:drop-shadow(0 3px 4px #10251b); }
    .ground-name { background:#102e24; color:#fff; border:1px solid #678c6e; border-radius:7px; padding:6px 8px; width:100%; box-sizing:border-box; font-size:12px; font-weight:600; line-height:1.35; overflow-wrap:anywhere; }
    @media(max-width:600px) { .ground-stadium { padding:28px 20px; border-width:5px; border-radius:60px; } .ground-player { width:108px; } .ground-name { font-size:11px; } .ground-players { gap:14px 8px; } }
    @media(prefers-reduced-motion:reduce) { .ground-player { animation:none; } }
    </style>
    """ + '<div class="ground-toolbar"><span id="layout-status" aria-live="polite">Drag players to arrange your field</span><button id="reset" type="button">Reset positions</button></div><div class="ground-stadium" role="region" aria-label="Draggable playing eleven">' + ''.join(rows) + '</div>' + """
    <script>
    const field = document.querySelector('.ground-stadium');
    const players = Array.from(field.querySelectorAll('.ground-player'));
    const status = document.querySelector('#layout-status');
    const storageKey = 'xi-field-LAYOUT_KEY';
    let saved = {};
    try { saved = JSON.parse(sessionStorage.getItem(storageKey) || '{}') || {}; } catch (_) {}
    const bounds = field.getBoundingClientRect();
    // Capture the role-based positions before converting to free placement.
    const defaults = players.map(p => {
        const r = p.getBoundingClientRect();
        return {x:(r.left-bounds.left-field.clientLeft+r.width/2)/field.clientWidth,
                y:(r.top-bounds.top-field.clientTop+r.height/2)/field.clientHeight};
    });
    field.style.height = field.clientHeight + 'px';
    field.style.boxSizing = 'content-box';
    field.style.padding = '0';
    const positions = {};
    function place(p, point) {
        const mx = p.offsetWidth/2/field.clientWidth + .025;
        const my = p.offsetHeight/2/field.clientHeight + .025;
        const v = {x:Math.max(mx,Math.min(1-mx,point.x)), y:Math.max(my,Math.min(1-my,point.y))};
        p.style.left = (v.x*100)+'%'; p.style.top = (v.y*100)+'%';
        positions[p.dataset.player] = v;
    }
    function save() {
        try { sessionStorage.setItem(storageKey, JSON.stringify(positions)); status.textContent='Layout saved in this browser tab'; }
        catch (_) { status.textContent='Layout updated (available until this view reloads)'; }
    }
    players.forEach((p,i) => {
        const name = p.querySelector('.ground-name').textContent;
        p.dataset.player = name; p.tabIndex=0; p.setAttribute('role','button');
        p.setAttribute('aria-label',name + ': drag or use arrow keys to move');
        p.style.position='absolute'; p.style.transform='translate(-50%,-50%)'; p.style.animation='none';
        p.style.width='clamp(70px, 17%, 130px)';
        field.appendChild(p);
        const s=saved[name];
        place(p,s && Number.isFinite(s.x) && Number.isFinite(s.y) ? s : defaults[i]);
        let drag=null;
        p.addEventListener('pointerdown',e=>{
            if(e.button!==0) return;
            e.preventDefault(); p.focus({preventScroll:true});
            drag={id:e.pointerId,x:e.clientX,y:e.clientY,start:{...positions[name]}};
            p.setPointerCapture(e.pointerId); p.classList.add('dragging');
        });
        p.addEventListener('pointermove',e=>{
            if(!drag || e.pointerId!==drag.id) return;
            place(p,{x:drag.start.x+(e.clientX-drag.x)/field.clientWidth,y:drag.start.y+(e.clientY-drag.y)/field.clientHeight});
        });
        const finish=()=>{if(!drag)return; drag=null;p.classList.remove('dragging');save();};
        p.addEventListener('pointerup',finish); p.addEventListener('pointercancel',finish); p.addEventListener('lostpointercapture',finish);
        p.addEventListener('keydown',e=>{
            const delta={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];
            if(!delta)return;
            e.preventDefault();const s=positions[name], step=e.shiftKey?.04:.01;
            place(p,{x:s.x+delta[0]*step,y:s.y+delta[1]*step});save();
        });
    });
    field.querySelectorAll('.ground-role').forEach(r=>r.remove());
    document.querySelector('#reset').addEventListener('click',()=>{
        players.forEach((p,i)=>place(p,defaults[i]));save();status.textContent='Role-based positions restored';
    });
    new ResizeObserver(()=>players.forEach(p=>place(p,positions[p.dataset.player]))).observe(field);
    </script>
    """.replace('LAYOUT_KEY', layout_key), height=950, scrolling=True)
