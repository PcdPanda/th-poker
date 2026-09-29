"use strict";

// The browser side of the local web table: it renders what /api returns and sends actions.
// All text goes into the page with textContent, so nothing the server sends becomes markup.

const SUITS = { c: "♣", d: "♦", h: "♥", s: "♠" };
const SUIT_NAMES = { c: "clubs", d: "diamonds", h: "hearts", s: "spades" };
const RANK_NAMES = {
  A: "ace", K: "king", Q: "queen", J: "jack", T: "ten", 9: "nine", 8: "eight", 7: "seven",
  6: "six", 5: "five", 4: "four", 3: "three", 2: "two",
};
const RANKS = "AKQJT98765432";
const BADGES = { BTN: "Dealer", SB: "Small blind", BB: "Big blind" };
const POSITIONS = { HJ: "hijack", CO: "cutoff", BTN: "dealer", SB: "small blind", BB: "big blind" };
const DIFFICULTY = { 1: "Easy", 2: "Medium", 3: "Hard" };
const STREETS = { preflop: "Before the flop", flop: "Flop", turn: "Turn", river: "River" };
const PRESETS = [["⅓ pot", 1 / 3], ["½ pot", 1 / 2], ["⅔ pot", 2 / 3], ["Pot", 1]];
const CARD = /^[2-9TJQKA][cdhs]$/;
const HELPERS_KEY = "thpoker.hideHelpers";
let session = null;
let state = null;
let busy = false;

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  let response;
  let data;
  try {
    response = await fetch("/api/" + path, options);
    data = await response.json();
  } catch (error) {
    throw new Error("Can't reach the table. Is `thpoker web` still running?");
  }
  if (!response.ok) {
    const text = data.error || response.statusText;
    throw new Error(text.charAt(0).toUpperCase() + text.slice(1) + ".");
  }
  return data;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function card(text, small) {
  const node = element("span", "card suit-" + text[1] + (small ? " small" : ""));
  node.textContent = (text[0] === "T" ? "10" : text[0]) + SUITS[text[1]];
  node.setAttribute("aria-label", RANK_NAMES[text[0]] + " of " + SUIT_NAMES[text[1]]);
  return node;
}

function withCards(line, className) {
  // A line of text with every card code ("As", "Th") drawn as a small card.
  const node = element("div", className);
  for (const [index, word] of line.split(" ").entries()) {
    if (index) node.append(" ");
    const bare = word.replace(/[.,:;]$/, "");
    if (CARD.test(bare)) {
      node.append(card(bare, true));
      if (bare !== word) node.append(word.slice(bare.length));
    } else {
      node.append(word);
    }
  }
  return node;
}

function prettyLine(line, className) {
  // The coach's text uses suit symbols ("A♠ 10♥"); draw those as small cards too.
  const node = element("div", className);
  const parts = line.split(/((?:10|[2-9JQKA])[♣♦♥♠])/);
  for (const part of parts) {
    const match = /^(10|[2-9JQKA])([♣♦♥♠])$/.exec(part);
    if (match) {
      const suit = Object.keys(SUITS).find((key) => SUITS[key] === match[2]);
      node.append(card((match[1] === "10" ? "T" : match[1]) + suit, true));
    } else if (part) {
      node.append(part);
    }
  }
  return node;
}

function button(label, handler, className, key) {
  const node = element("button", className || "", label);
  node.type = "button";
  node.addEventListener("click", handler);
  if (key) node.dataset.key = key;
  return node;
}

function chips(value) {
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
}

function signed(value) {
  return (value > 0 ? "+" : value < 0 ? "−" : "") + chips(Math.abs(value));
}

function percent(share) {
  return Math.round(100 * share) + "%";
}

function showError(id, error) {
  document.getElementById(id).textContent = error ? error.message : "";
}

async function run(id, work) {
  // One request at a time: buttons are disabled until it answers, and an answer that arrives
  // after a new game has started is dropped.
  if (busy) return;
  busy = true;
  document.body.classList.add("busy");
  showError(id, null);
  const mine = session;
  try {
    await work(() => mine === session);
  } catch (error) {
    showError(id, error);
  } finally {
    busy = false;
    document.body.classList.remove("busy");
  }
}

function show(view) {
  for (const id of ["setup", "table", "history"]) document.getElementById(id).hidden = id !== view;
  document.getElementById("new-game").hidden = view !== "table";
  document.getElementById("helpers-toggle").hidden = view !== "table";
}

// The table

function hudText(hud) {
  if (hud.hands < state.hud_min_hands) return "Watching: " + hud.hands + " of " + state.hud_min_hands + " hands";
  return "Plays " + percent(hud.plays) + " of hands · raises " + percent(hud.raises);
}

function hudTip(hud) {
  if (hud.hands < state.hud_min_hands) return "After " + state.hud_min_hands + " hands these numbers start to mean something.";
  const parts = [
    "Plays " + percent(hud.plays) + " of hands: how often it puts chips in before the flop by choice.",
    "Raises " + percent(hud.raises) + ": how often it raises before the flop.",
  ];
  if (hud.reraises !== null) parts.push("Re-raises " + percent(hud.reraises) + " of the times it faces a raise.");
  if (hud.aggression !== null) parts.push("After the flop it bets or raises " + hud.aggression.toFixed(1) + " times for every call.");
  return parts.join("\n");
}

function renderSeat(seat, mine) {
  const out = seat.dealt_in === false || seat.folded;
  const box = element("div", "seat" + (mine ? " you" : "") + (out ? " out" : ""));
  const head = element("div", "head");
  head.append(element("span", "name", seat.name));
  if (seat.position && BADGES[seat.position]) {
    const heads = state.seats.filter((s) => s.dealt_in).length === 2 && seat.position === "BTN";
    head.append(element("span", "badge", heads ? "Dealer · Small blind" : BADGES[seat.position]));
  }
  if (seat.style) {
    const tag = element("span", "tag helper", seat.style);
    tag.title = seat.tip;
    head.append(tag);
  }
  box.append(head);
  const line = element("div", "chips");
  line.append(element("span", "stack", chips(seat.stack) + " chips"));
  if (seat.committed) line.append(element("span", "bet", "Bet " + chips(seat.committed)));
  box.append(line);
  if (seat.in_pot) box.append(element("div", "in-pot helper", "In the pot this hand: " + chips(seat.in_pot)));
  const status = seat.folded ? "Folded" : seat.all_in ? "All-in" : seat.last_action ? seat.last_action.charAt(0).toUpperCase() + seat.last_action.slice(1) : "";
  if (status) box.append(element("div", "last", status));
  const cards = element("div", "cards");
  for (const text of seat.cards || []) cards.append(card(text));
  if (!mine && seat.dealt_in && !seat.folded && !seat.cards && state.board !== undefined) {
    cards.append(element("span", "card back"), element("span", "card back"));
  }
  box.append(cards);
  if (mine && state.your_hand) box.append(element("div", "hand-name helper", state.your_hand));
  if (seat.hud) {
    const hud = element("div", "hud helper", hudText(seat.hud));
    hud.title = hudTip(seat.hud);
    box.append(hud);
  }
  return box;
}

function you() {
  return state.seats.find((seat) => seat.is_user);
}

function renderTable() {
  const opponents = document.getElementById("opponents");
  opponents.replaceChildren(...state.seats.filter((s) => !s.is_user).map((s) => renderSeat(s, false)));
  document.getElementById("me").replaceChildren(renderSeat(you(), true));
  const board = document.getElementById("board");
  board.replaceChildren();
  if (state.board !== undefined) {
    for (const text of state.board) board.append(card(text));
    for (let i = state.board.length; i < 5; i++) board.append(element("span", "card empty"));
  }
  document.getElementById("pot").textContent = state.pot === undefined ? "" : "Pot " + chips(state.pot);
}

function statusText() {
  if (state.your_turn) {
    const call = state.legal.call;
    return call !== null ? "Your turn: " + chips(call) + " to call." : "Your turn: nobody has bet, so you can check.";
  }
  if (state.session_over) return "The session is over.";
  if (state.awaiting_rebuy) return "You are out of chips.";
  if (state.knocked_out) return "You are out of the tournament.";
  if (state.hand_over && state.hand_result !== undefined) {
    const net = state.hand_result;
    const hand = net > 0 ? "you won " + chips(net) : net < 0 ? "you lost " + chips(-net) : "you broke even";
    return "Hand over: " + hand + " this hand.";
  }
  return state.hand_over ? "Hand over." : "Waiting for the others.";
}

function renderStatus() {
  document.getElementById("status").textContent = statusText();
  const odds = document.getElementById("odds");
  odds.textContent = "";
  if (state.your_turn && state.call_needs) {
    odds.textContent = "Pot odds: call " + chips(state.legal.call) + " into a pot of " + chips(state.pot)
      + ", so calling pays if you win more than " + percent(state.call_needs) + " of the time.";
  }
}

function raiseLabel(kind, amount) {
  return (kind === "bet" ? "Bet " : "Raise to ") + chips(amount);
}

function renderSizing(raise, raiseButton) {
  const sizing = document.getElementById("sizing");
  sizing.hidden = !raise;
  if (!raise) return;
  const amount = document.getElementById("amount");
  const slider = document.getElementById("slider");
  for (const input of [amount, slider]) {
    input.min = raise.min;
    input.max = raise.max;
    input.step = state.step;
  }
  const set = (value) => {
    amount.value = slider.value = value;
    const allIn = Number(value) >= raise.max;
    raiseButton.textContent = raiseLabel(raise.kind, Number(value)) + (allIn ? " (all-in)" : "");
  };
  slider.oninput = () => set(slider.value);
  amount.oninput = () => set(amount.value);
  set(raise.min);
  const presets = document.getElementById("presets");
  presets.replaceChildren();
  const call = state.legal.call || 0;
  const mine = you().committed || 0;
  for (const [label, share] of PRESETS) {
    // A pot-fraction bet or raise: call first, then add that share of the pot after calling.
    const target = mine + call + share * (state.pot + call);
    // Round to the smallest chip (a 0.01 unit with 0.5/1 blinds), then trim float noise.
    const rounded = Number((Math.round(target / state.step) * state.step).toFixed(6));
    presets.append(button(label, () => set(Math.min(raise.max, Math.max(raise.min, rounded))), "chip"));
  }
  presets.append(button("All-in", () => set(raise.max), "chip all-in"));
}

function renderActions() {
  const actions = document.getElementById("actions");
  const between = document.getElementById("between");
  actions.replaceChildren();
  between.replaceChildren();
  if (state.your_turn) {
    const legal = state.legal;
    if (legal.fold && !legal.check) actions.append(button("Fold", () => act({ kind: "fold" }), "neutral", "f"));
    if (legal.check) actions.append(button("Check", () => act({ kind: "check" }), "primary", "k"));
    if (legal.call !== null) {
      const allIn = legal.call >= you().stack ? " (all-in)" : "";
      actions.append(button("Call " + chips(legal.call) + allIn, () => act({ kind: "call" }), "primary", "c"));
    }
    let raiseButton = null;
    if (legal.raise) {
      raiseButton = button("", () => act({ kind: legal.raise.kind, amount: Number(document.getElementById("amount").value) }), "raise", "r");
      actions.append(raiseButton);
    }
    if (state.coach) actions.append(button("Ask the coach", hint, "secondary", "h"));
    renderSizing(legal.raise, raiseButton);
  } else {
    renderSizing(null, null);
  }
  const check = state.coach && state.acted ? button("Check my last move", analyze, "secondary") : null;
  if (check && !state.hand_over) actions.append(check);
  if (state.hand_over) {
    if (state.session_over) {
      between.append(element("span", "", "The session is over. "));
    } else if (state.awaiting_rebuy) {
      between.append(button("Rebuy", rebuy, "primary"));
    } else if (state.knocked_out) {
      between.append(button("Fast-forward to the end", next, "primary", "n"));
    } else {
      between.append(button("Next hand", next, "primary", "n"));
    }
    between.append(button("Review this hand", review, "secondary"));
    if (check) between.append(check);
  }
}

function renderLog() {
  const log = document.getElementById("log");
  log.replaceChildren();
  for (const line of state.log) {
    if (line.startsWith("=== ")) {
      log.append(element("div", "log-hand", line.slice(4)));
    } else if (line.startsWith("--- ")) {
      const [street, cards] = line.slice(4).split(": ");
      const node = withCards(street + ": " + (cards || ""), "log-street");
      log.append(node);
    } else if (line.startsWith("*** ")) {
      log.append(element("div", "log-event", line.slice(4)));
    } else {
      log.append(withCards(line.trim(), "log-line"));
    }
  }
  log.scrollTop = log.scrollHeight;
}

function handsTable(rows, reviewer) {
  if (!rows.length) return element("p", "", "No finished hands yet.");
  const table = element("table", "hands");
  const head = element("tr");
  for (const title of ["Hand", "Seat", "Your cards", "Board", "Put in", "Result", ""]) head.append(element("th", "", title));
  table.append(head);
  for (const row of rows.slice().reverse()) {
    const tr = element("tr");
    tr.append(element("td", "", row.hand), element("td", "", POSITIONS[row.position] || "early"));
    const cards = element("td");
    for (const text of row.cards.split(" ").filter(Boolean)) cards.append(card(text, true));
    const board = element("td");
    for (const text of row.board.split(" ").filter(Boolean)) board.append(card(text, true));
    tr.append(cards, board, element("td", "", chips(row.put_in)));
    tr.append(element("td", row.result > 0 ? "up" : row.result < 0 ? "down" : "", signed(row.result)));
    const action = element("td");
    if (reviewer) action.append(button("Review", () => reviewer(row.hand), "small"));
    tr.append(action);
    table.append(tr);
  }
  return table;
}

function renderHands() {
  const net = document.getElementById("net");
  net.textContent = state.session_net === undefined ? "" : "· since you sat down: " + signed(state.session_net);
  document.getElementById("hands-table").replaceChildren(handsTable(state.hands, null));
  document.getElementById("csv").href = "/api/sessions/" + session + "/hands.csv";
}

function render() {
  renderTable();
  renderStatus();
  renderActions();
  renderLog();
  renderHands();
}

function clearCoach() {
  document.getElementById("coach").replaceChildren();
  document.getElementById("coach-panel").hidden = true;
}

async function act(action) {
  await run("table-error", async (current) => {
    const answer = await api("sessions/" + session + "/action", action);
    if (!current()) return;
    state = answer;
    clearCoach();
    render();
  });
}

async function next() {
  await run("table-error", async (current) => {
    const answer = await api("sessions/" + session + "/next", {});
    if (!current()) return;
    state = answer;
    clearCoach();
    render();
  });
}

async function rebuy() {
  await run("table-error", async (current) => {
    const answer = await api("sessions/" + session + "/rebuy", {});
    if (!current()) return;
    state = answer;
    render();
  });
}

function newGame() {
  if (!window.confirm("Leave this game and start a new one?")) return;
  session = null;
  state = null;
  clearCoach();
  document.getElementById("table-error").textContent = "";
  show("setup");
}

async function coach(path, waiting) {
  await run("table-error", async (current) => {
    const panel = document.getElementById("coach-panel");
    const target = document.getElementById("coach");
    panel.hidden = false;
    target.replaceChildren(element("p", "waiting", waiting));
    try {
      const answer = await api("sessions/" + session + "/" + path, {});
      if (current()) target.replaceChildren(coachView(answer, path === "review"));
    } catch (error) {
      target.replaceChildren();
      panel.hidden = true;
      throw error;
    }
  });
}

function hint() {
  return coach("hint", "Thinking...");
}

function analyze() {
  return coach("analyze", "Checking your last move (a second or two)...");
}

function review() {
  return coach("review", "Reviewing the hand (a few seconds)...");
}

// The coach's answers

function optionsTable(decision) {
  const table = element("table", "options");
  const head = element("tr");
  const unit = decision.unit === "chips" ? "" : " (" + decision.unit + ")";
  for (const title of ["Option", "A strong player does it", "Average result vs these opponents" + unit, "vs a strong player" + unit]) {
    head.append(element("th", "", title));
  }
  table.append(head);
  const value = (worth, noise) => {
    const cell = element("td", "", decision.unit === "chips" ? signed(worth) : worth.toFixed(2) + "%");
    if (noise) cell.title = "± " + chips(noise) + ": an estimate from sampling";
    return cell;
  };
  for (const option of decision.options) {
    const tr = element("tr", option.best ? "best" : "");
    const name = element("td", "", option.action);
    if (option.chosen) name.append(element("span", "mark", "your move"));
    if (option.best) name.append(element("span", "mark best", "best"));
    const share = element("td", "share");
    const bar = element("span", "bar");
    bar.style.width = Math.round(100 * option.strong_share) + "%";
    share.append(bar, element("span", "", percent(option.strong_share)));
    tr.append(name, share, value(option.vs_bots, option.vs_bots_noise), value(option.vs_strong, option.vs_strong_noise));
    table.append(tr);
  }
  return table;
}

function rangeGrid(range) {
  const grid = element("div", "grid");
  grid.setAttribute("role", "img");
  grid.setAttribute("aria-label", range.name + " likely holds about " + percent(range.width) + " of hands");
  for (let row = 0; row < 13; row++) {
    for (let column = 0; column < 13; column++) {
      const a = RANKS[Math.min(row, column)];
      const b = RANKS[Math.max(row, column)];
      const label = row === column ? a + b : a + b + (row < column ? "s" : "o");
      const weight = range.classes[row * 13 + column];
      const cell = element("span", "cell", label);
      cell.style.setProperty("--w", String(weight));
      cell.title = label + ": " + percent(weight);
      grid.append(cell);
    }
  }
  return grid;
}

function decisionView(decision, open) {
  const box = element("article", "decision verdict-" + (decision.verdict || "hint"));
  const title = element("div", "decision-title");
  title.append(element("span", "street", STREETS[decision.street] + ":"));
  for (const text of decision.hole) title.append(card(text, true));
  if (decision.board.length) {
    title.append(element("span", "on", "on"));
    for (const text of decision.board) title.append(card(text, true));
  }
  if (decision.chosen) title.append(element("span", "chose", "you chose " + decision.chosen));
  box.append(title, element("p", "headline", decision.headline));
  for (const line of decision.summary) box.append(element("p", "summary", line));
  box.append(optionsTable(decision));
  const details = element("details");
  details.open = open;
  details.append(element("summary", "", "Show details"));
  const list = element("ul");
  for (const line of decision.details) list.append(element("li", "", line));
  details.append(list);
  for (const range of decision.ranges) {
    if (range.width >= 0.6) continue;  // nearly every hand: the grid would be solid colour
    details.append(element("p", "grid-title", range.name + "'s likely hands (brighter means more likely):"), rangeGrid(range));
  }
  box.append(details);
  return box;
}

function copyButton(text) {
  const node = button("Copy for AI chat", async () => {
    try {
      await navigator.clipboard.writeText(text);
      node.textContent = "Copied: paste it into a chat assistant";
    } catch (error) {
      const box = element("textarea", "copy-box");
      box.readOnly = true;
      box.value = text;
      node.replaceWith(element("p", "", "Copying was blocked; select this text and copy it:"), box);
      box.select();
    }
  }, "secondary");
  node.title = "Copies the whole hand and this analysis as plain text";
  return node;
}

function coachView(answer, full) {
  const view = element("div", "coach-view");
  view.append(copyButton(answer.copy_text));
  const story = element("details", "story");
  story.open = full;
  story.append(element("summary", "", full ? "The hand" : "The hand so far"));
  for (const line of answer.history) story.append(prettyLine(line, line.startsWith(" ") ? "story-line" : "story-head"));
  view.append(story);
  if (!answer.decisions.length) view.append(element("p", "", "You made no decision in that hand."));
  for (const decision of answer.decisions) view.append(decisionView(decision, false));
  if (answer.result) view.append(element("p", "result", answer.result));
  return view;
}

// Past sessions

async function openHistory() {
  show("history");
  document.getElementById("session-hands").replaceChildren();
  document.getElementById("history-coach-panel").hidden = true;
  await run("history-error", async () => {
    const answer = await api("history");
    const target = document.getElementById("sessions");
    if (!answer.logging) {
      target.replaceChildren(element("p", "", "Sessions aren't saved: the table was started with --no-log."));
      return;
    }
    if (!answer.sessions.length) {
      target.replaceChildren(element("p", "", "No saved sessions yet. Play a game and it will show up here."));
      return;
    }
    const table = element("table", "hands");
    const head = element("tr");
    for (const title of ["Date", "Game", "Players", "Opponents", "Hands", "Result", ""]) head.append(element("th", "", title));
    table.append(head);
    for (const saved of answer.sessions) {
      const tr = element("tr");
      tr.append(element("td", "", saved.date), element("td", "", saved.mode === "cash" ? "Cash game" : "Tournament"));
      tr.append(element("td", "", saved.players), element("td", "", DIFFICULTY[saved.difficulty]), element("td", "", saved.hands));
      tr.append(element("td", saved.net > 0 ? "up" : saved.net < 0 ? "down" : "", signed(saved.net)));
      const open = element("td");
      const label = (saved.mode === "cash" ? "Cash game" : "Tournament") + " on " + saved.date;
      open.append(button("Open", () => openSession(saved.name, label), "small"));
      tr.append(open);
      table.append(tr);
    }
    target.replaceChildren(table);
  });
}

async function openSession(name, label) {
  await run("history-error", async () => {
    const answer = await api("history/" + encodeURIComponent(name));
    const target = document.getElementById("session-hands");
    const link = element("a", "", "Download this session as CSV");
    link.href = "/api/history/" + encodeURIComponent(name) + "/hands.csv";
    target.replaceChildren(element("h2", "", "Hands: " + label), link, handsTable(answer.rows, (hand) => reviewSaved(name, hand)));
  });
}

async function reviewSaved(name, hand) {
  await run("history-error", async () => {
    const panel = document.getElementById("history-coach-panel");
    const target = document.getElementById("history-coach");
    panel.hidden = false;
    target.replaceChildren(element("p", "waiting", "Reviewing hand " + hand + " (a few seconds)..."));
    try {
      const answer = await api("history/" + encodeURIComponent(name) + "/review", { hand });
      target.replaceChildren(coachView(answer, true));
      panel.scrollIntoView({ behavior: "smooth" });
    } catch (error) {
      target.replaceChildren();
      panel.hidden = true;
      throw error;
    }
  });
}

// Setup and page-wide controls

function modeChanged() {
  const tournament = document.querySelector("input[name=mode]:checked").value === "tournament";
  document.getElementById("blinds-field").hidden = tournament;
  document.querySelector("input[name=coach]").checked = !tournament;  // DESIGN 7.8: off in tournaments
}

for (const radio of document.querySelectorAll("input[name=mode]")) radio.addEventListener("change", modeChanged);

document.getElementById("setup-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  const options = {
    mode: form.get("mode"),
    seats: Number(form.get("seats")),
    tier: Number(form.get("tier")),
    coach: form.get("coach") === "on",
  };
  if (options.mode === "cash") options.blinds = form.get("blinds");
  if (form.get("stack")) options.stack = Number(form.get("stack"));
  run("setup-error", async () => {
    const created = await api("sessions", options);
    session = created.id;
    state = created.state;
    clearCoach();
    show("table");
    render();
  });
});

document.getElementById("new-game").addEventListener("click", newGame);
document.getElementById("open-history").addEventListener("click", openHistory);
document.getElementById("history-back").addEventListener("click", () => show("setup"));

const hideHelpers = document.getElementById("hide-helpers");
hideHelpers.checked = window.localStorage.getItem(HELPERS_KEY) === "1";
document.body.classList.toggle("no-helpers", hideHelpers.checked);
hideHelpers.addEventListener("change", () => {
  window.localStorage.setItem(HELPERS_KEY, hideHelpers.checked ? "1" : "0");
  document.body.classList.toggle("no-helpers", hideHelpers.checked);
});

document.addEventListener("keydown", (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey || document.getElementById("table").hidden) return;
  if (!/^[a-z]$/i.test(event.key) || event.target.closest("input, select, textarea")) return;
  const target = document.querySelector('#table button[data-key="' + event.key.toLowerCase() + '"]');
  if (target && !busy) {
    event.preventDefault();
    target.click();
  }
});
