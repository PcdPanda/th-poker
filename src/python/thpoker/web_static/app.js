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
const BADGES = { SB: "small blind", BB: "big blind" };
const POSITIONS = { HJ: "hijack", CO: "cutoff", BTN: "dealer", SB: "small blind", BB: "big blind" };
const SEAT_WORDS = {
  UTG: "First to act", "UTG+1": "Second to act", "UTG+2": "Third to act", HJ: "Hijack",
  CO: "Cutoff", BTN: "Dealer", SB: "Small blind", BB: "Big blind",
};
const DIFFICULTY = { 1: "Easy", 2: "Medium", 3: "Hard", 4: "Expert" };
const DIFFICULTY_WORDS = {
  1: "Plays by simple rules about its own cards. Easy to read and easy to beat.",
  2: "Weighs its chance to win against the cost of calling, and guesses your cards from how you bet.",
  3: "Plays like a strong player: standard charts before the flop, and a hard-to-read mix of bets and bluffs after it.",
  4: "Plays like Hard at first, then adjusts to your habits the longer you play. Only this session counts.",
};
const STREETS = { preflop: "Before the flop", flop: "Flop", turn: "Turn", river: "River" };
const PRESETS = [["⅓ pot", 1 / 3], ["½ pot", 1 / 2], ["⅔ pot", 2 / 3], ["Pot", 1]];
const CARD = /^[2-9TJQKA][cdhs]$/;
const HELPERS_KEY = "thpoker.hideHelpers";
const HELPERS_TIP = "Hide pot odds, hand names, chips-in-the-pot totals and player stats, as at a real table";
const CHANCE_KEY = "thpoker.showChance";
const SVG = "http://www.w3.org/2000/svg";
const ICONS = {
  copy: ["M9 9h11v11H9z", "M5 15H4V4h11v1"],
  chat: ["M4 4h16v11H9l-5 4z", "M8 9h8", "M8 12h5"],
  download: ["M12 4v11", "M7 10l5 5 5-5", "M4 20h16"],
  done: ["M5 12l5 5 9-10"],
};
let session = null;
let state = null;
let busy = false;
let chanceAsked = 0;
let chance = null;
let guess = null;
let revealed = false;
const openMoves = new Set(); // "session/hand" of the rows showing their moves, across re-renders
const latestRows = new Map();  // each shown hand's row as last read, by "session/hand"
let refreshTimer = null;
let refreshes = 0;  // counts what the hands tables show, so a re-read for an older view is dropped
let handsOrder = null;  // the hands tables' sort as { column, up }, or null for the newest hand first
const SEAT_ORDER = Object.keys(SEAT_WORDS);
const RATING_TIP = "Your moves this hand, rated 0 to 1, with moves in bigger pots counting for more: 1 means you picked the best option every time";
const SOLVER_TIP = "Show the coach's numbers for every decision, in hand reviews and copies";
const GOD_TIP = "Also show the bots' styles and cards, and the numbers knowing them (finished hands)";
const HIDDEN_TIP = ". This shows the hidden styles.";
// What hand reviews and copies show, for the whole page: off at every visit, and never stored.
const switches = { solver: false, god: false };
let coachShown = null;  // the coach panel's answer and whether it reviews a whole hand
const reviews = new Map();  // each opened hand row's full review by "session/hand"
const loading = new Set();  // the rows whose review is on its way
let inBB = false;  // the hands tables' Put in and Result in big blinds rather than chips
let clock = null;  // the move timer of this turn: { turn, deadline, warned, fired }, or null
let clockPaused = null;  // when a coach request began: waiting on it isn't the player's time
let controlsAbove = false;
let timeoutNote = "";  // what the timer did, shown with the status until the next move
// Title, tip, class, and the value a click on the title sorts by, and whether it is an amount.
const COLUMNS = [
  ["Hand", null, null, (row) => row.hand],
  ["Seat", null, null, (row) => SEAT_ORDER.indexOf(row.position)],
  ["Cards"],
  ["Top %", "Your two starting cards are in this top share of all starting hands: lower is stronger, and aces are the top 0.5%", "number", (row) => row.hand_rank],
  ["Board", null, "wide-only"],
  ["Chance", "Your chance to win at showdown at your last move, reading the others' hands from their play the way a strong player would, so it can differ from the chance shown during play", "number", (row) => row.win_chance],
  ["Put in", null, "wide-only number", (row) => amount(row.put_in, row)],
  ["Result", null, "number", (row) => amount(row.result, row)],
  ["Rating", RATING_TIP, "number", (row) => row.rating],
];

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

function icon(name) {
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", "icon");
  for (const d of ICONS[name]) {
    const path = document.createElementNS(SVG, "path");
    path.setAttribute("d", d);
    svg.append(path);
  }
  return svg;
}

function copyIcon(name, tip, text) {
  // Copies `text`, or the text a function fetches; the icon turns into a tick for a moment.
  // If the browser refuses the clipboard, the text appears in a box at the foot of the panel
  // to copy by hand.
  const node = element("button", "icon-button");
  node.type = "button";
  node.title = tip;
  node.setAttribute("aria-label", tip);
  node.append(icon(name));
  node.addEventListener("click", async () => {
    if (node.classList.contains("busy")) return;
    const panel = node.closest(".copy-host");
    const pending = Promise.resolve(typeof text === "function" ? text() : text);
    node.classList.add("busy");
    try {
      try {
        // Started in the click itself, so a browser that ties copying to the click (Safari)
        // still copies text that arrives a moment later.
        const blob = pending.then((value) => new Blob([value], { type: "text/plain" }));
        await navigator.clipboard.write([new ClipboardItem({ "text/plain": blob })]);
      } catch (error) {
        await navigator.clipboard.writeText(await pending);
      }
      node.replaceChildren(icon("done"));
      setTimeout(() => node.replaceChildren(icon(name)), 1500);
    } catch (error) {
      const box = element("textarea", "copy-box");
      box.readOnly = true;
      box.value = await pending.catch(() => "");
      const fallback = element("div", "copy-fallback");
      const words = box.value ? "Copying was blocked; select this text and copy it:" : "Couldn't get the hand to copy. Try again.";
      fallback.append(element("p", "", words));
      if (box.value) fallback.append(box);
      panel.querySelector(".copy-fallback")?.remove();
      panel.append(fallback);
      if (box.value) box.select();
    } finally {
      node.classList.remove("busy");
    }
  });
  return node;
}

function copyMode() {
  if (switches.solver) return switches.god ? "solver_god" : "solver";
  return switches.god ? "god" : "plain";
}

function copyTip(hand) {
  const on = (flag) => (flag ? "on" : "off");
  return "Copy hand " + hand + " for an AI chat (solver " + on(switches.solver) + ", God's view " + on(switches.god) + ")";
}

function makeSwitch(name, text, tip, checked, changed) {
  const label = element("label", "switch");
  label.title = tip;
  const input = element("input");
  input.type = "checkbox";
  input.name = name;
  input.setAttribute("role", "switch");
  input.checked = checked;
  if (changed) input.addEventListener("change", () => changed(input));
  label.append(text, input);
  return label;
}

function reviewSwitches() {
  // The Solver and God's view switches, which hold for the whole page and stay in view while
  // scrolling, and room for a copy icon.
  const group = element("div", "review-switches");
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "Reviews and copies");
  group.append(
    element("span", "", "Reviews and copies:"),
    makeSwitch("solver", "Solver", SOLVER_TIP, switches.solver, switchChanged),
    makeSwitch("god", "God's view", GOD_TIP, switches.god, switchChanged),
    element("span", "tools"),
  );
  return group;
}

function renderSwitches() {
  // The table's copy icon copies the hand shown, as the switches set it.
  const host = document.querySelector("#table .copy-host");
  host.querySelector(".copy-fallback")?.remove();
  const target = host.querySelector(".tools");
  target.replaceChildren();
  if (state.hand_number) {
    const path = "sessions/" + session + "/hands/" + state.hand_number + "/copy/";
    target.append(copyIcon("copy", copyTip(state.hand_number), () => api(path + copyMode()).then((answer) => answer.text)));
  }
  document.getElementById("god-later").hidden = !(switches.god && state.hand_number && !state.hand_over);
  const hidden = state.seats.some((seat) => !seat.is_user && !seat.style);
  host.querySelector("input[name=god]").parentElement.title = GOD_TIP + (hidden ? HIDDEN_TIP : "");
}

function renderHistoryCopy() {
  const target = tools("history-tools");
  if (!historyShown) return;
  target.append(copyIcon("copy", copyTip(historyShown.hand), () => historyShown.answer.texts[copyMode()]));
  const hidden = historyShown.answer.hides_styles ? ". This shows the hidden styles." : "";
  document.querySelector("#history .god-switch").title = GOD_TIP + hidden;
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

function storyLine(line) {
  // Indented lines are the moves of the step above them; a line ending in ":" starts a step.
  const text = line.trim();
  return prettyLine(text, line.startsWith(" ") ? "story-line" : text.endsWith(":") ? "story-head step" : "story-head");
}

function button(label, handler, className, key) {
  const node = element("button", className || "", label);
  node.type = "button";
  node.addEventListener("click", handler);
  if (key) {
    node.dataset.key = key;
    node.title = "Key: " + key.toUpperCase();
  }
  return node;
}

function chips(value) {
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
}

function signed(value) {
  return (value > 0 ? "+" : value < 0 ? "−" : "") + chips(Math.abs(value));
}

function percent(share) {
  return (100 * share).toFixed(1) + "%";
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
  refreshRows(null, null, 0);
  for (const id of ["setup", "table", "history"]) document.getElementById(id).hidden = id !== view;
  document.getElementById("new-game").hidden = view !== "table";
  document.getElementById("helpers-toggle").hidden = view !== "table";
  document.getElementById("training-pill").hidden = view !== "table" || !state || !state.training;
  document.getElementById("chance-mode").hidden = view !== "table" || !state || !state.training;
  renderToTable();
}

function renderToTable() {
  // The way back up from the log and reviews below the table; it turns gold on the player's turn.
  const button = document.getElementById("to-table");
  button.hidden = !controlsAbove || document.getElementById("table").hidden;
  const turn = Boolean(state && state.your_turn);
  const left = turn && state.timer && clock ? secondsLeft() : null;
  button.classList.toggle("turn", turn);
  button.classList.toggle("low", left !== null && left <= Math.min(10, state.timer / 3));
  document.getElementById("to-table-text").textContent = !turn ? "Back to table"
    : left !== null ? "Your turn · " + left + " s" : "Your turn";
}

function seatNames(players) {
  // Positions in the order they act before the flop, as the server names them (charts.seat_names).
  if (players === 2) return ["BTN", "BB"];
  const others = players - 3;
  const late = others >= 2 ? ["HJ", "CO"] : others === 1 ? ["CO"] : [];
  const early = Array.from({ length: others - late.length }, (_, i) => (i ? "UTG+" + i : "UTG"));
  return [...early, ...late, "BTN", "SB", "BB"];
}

function seatWords(code, players) {
  if (players === 2 && code === "BTN") return "Dealer and small blind";
  return SEAT_WORDS[code] + " (" + code + ")";
}

function trainingText(training, players) {
  const parts = ["Training"];
  if (training.position) parts.push(seatWords(training.position, players));
  if (training.hands) parts.push(training.hands);
  return parts.join(" · ");
}

// The table

function hudTip(hud) {
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
  const box = element("div", "seat" + (mine ? " you" : "") + (out ? " out" : "") + (mine && state.your_turn ? " turn" : ""));
  const head = element("div", "head");
  head.append(element("span", "name", seat.name));
  const players = state.seats.filter((s) => s.dealt_in).length;
  if (seat.position === "BTN") {
    const dealer = element("span", "dealer", "D");
    dealer.title = players === 2 ? "Dealer, also the small blind" : "Dealer: acts last after the flop";
    head.append(dealer);
  }
  if (BADGES[seat.position]) head.append(element("span", "badge", BADGES[seat.position]));
  box.append(head);
  const cards = element("div", "cards");
  for (const text of seat.cards || []) cards.append(card(text));
  if (!mine && seat.dealt_in && !seat.folded && !seat.cards && state.board !== undefined) {
    cards.append(element("span", "card back"), element("span", "card back"));
  }
  box.append(cards);
  if (mine && state.your_hand) box.append(element("div", "hand-name helper", state.your_hand));
  const line = element("div", "chips");
  line.append(element("span", "stack", chips(seat.stack)));
  if (seat.committed) line.append(element("span", "bet", chips(seat.committed)));
  box.append(line);
  const status = seat.folded ? "Folded" : seat.all_in ? "All-in" : seat.last_action ? seat.last_action.charAt(0).toUpperCase() + seat.last_action.slice(1) : "";
  if (status) box.append(element("div", "last", status));
  const note = element("div", "note helper");
  if (seat.style) {
    const tag = element("span", "tag", seat.style);
    tag.title = seat.tip;
    note.append(tag);
  }
  if (seat.in_pot) note.append(element("span", "", "In pot " + chips(seat.in_pot)));
  if (seat.hud && seat.hud.hands >= state.hud_min_hands) {
    const hud = element("span", "", "Plays " + percent(seat.hud.plays) + " · raises " + percent(seat.hud.raises));
    hud.title = hudTip(seat.hud);
    note.append(hud);
  }
  if (note.childNodes.length) box.append(note);
  return box;
}

function you() {
  return state.seats.find((seat) => seat.is_user);
}

function ring(count) {
  // Points spaced evenly along the oval's edge from the bottom, clockwise, as fractions of its
  // half-width and half-height: equal angles would crowd the seats at the narrow ends.
  const felt = document.getElementById("felt");
  const ratio = felt.clientHeight / felt.clientWidth || 0.5;
  const steps = 720;
  const points = [[0, 1]];
  const lengths = [0];
  for (let i = 1; i <= steps; i++) {
    const angle = Math.PI / 2 + (2 * Math.PI * i) / steps;
    points.push([Math.cos(angle), Math.sin(angle)]);
    const [x, y] = points[i - 1];
    lengths.push(lengths[i - 1] + Math.hypot(points[i][0] - x, (points[i][1] - y) * ratio));
  }
  const result = [];
  for (let k = 0, i = 0; k < count; k++) {
    while (lengths[i] < (lengths[steps] * k) / count) i++;
    result.push(points[i]);
  }
  return result;
}

function renderTable() {
  // Seats sit round the oval in playing order, starting on your left, with you at the bottom.
  const seats = state.seats;
  const mine = seats.findIndex((seat) => seat.is_user);
  const places = ring(seats.length);
  const boxes = seats.map((seat, index) => {
    const box = renderSeat(seat, index === mine);
    box.dataset.seat = String(index);
    const [x, y] = places[(index - mine + seats.length) % seats.length];
    box.style.setProperty("--x", (50 + 50 * x).toFixed(2) + "%");
    box.style.setProperty("--y", (50 + 54 * y).toFixed(2) + "%");  // a little past the rail
    return box;
  });
  document.getElementById("seats").replaceChildren(...boxes);
  const board = document.getElementById("board");
  board.replaceChildren();
  if (state.board !== undefined) {
    for (const text of state.board) board.append(card(text));
    for (let i = state.board.length; i < 5; i++) board.append(element("span", "card empty"));
  }
  const pot = document.getElementById("pot");
  pot.replaceChildren();
  if (state.pot !== undefined) pot.append(element("span", "chip-stack", chips(state.pot)));
  pot.title = "The pot";
}

function statusText() {
  if (state.your_turn) {
    const call = state.legal.call;
    return call !== null ? "Your turn · " + chips(call) + " to call" : "Your turn";
  }
  if (state.session_over) return "The session is over";
  if (state.awaiting_rebuy) return "You are out of chips";
  if (state.knocked_out) return "You are out of the tournament";
  if (state.hand_over && state.hand_result !== undefined) {
    const net = state.hand_result;
    return net > 0 ? "You won " + chips(net) : net < 0 ? "You lost " + chips(-net) : "You broke even";
  }
  return state.hand_over ? "Hand over" : "Waiting for the others";
}

function renderStatus() {
  document.getElementById("status").textContent = (timeoutNote ? timeoutNote + " " : "") + statusText();
  document.body.classList.toggle("your-turn", Boolean(state.your_turn));
  const odds = document.getElementById("odds");
  odds.textContent = "";
  odds.title = "";
  if (state.your_turn && state.call_needs) {
    odds.textContent = "Calling needs " + percent(state.call_needs) + " to win";
    odds.title = "Pot odds: call " + chips(state.legal.call) + " into a pot of " + chips(state.pot)
      + ", so calling pays if you win more than " + percent(state.call_needs) + " of the time.";
  }
  if (state.training) document.getElementById("training-pill").textContent = trainingText(state.training, state.seats.length);
}

function chanceMode() {
  return document.querySelector("input[name=chance]:checked").value;
}

async function showChance() {
  // Either an answer for the start of the turn, or a guess is answered all at once; an answer that arrives
  // after the table has moved on is dropped.
  const mine = ++chanceAsked;
  chance = null;
  guess = null;
  revealed = false;
  drawChance();
  if (!state || !state.training || !state.your_turn || chanceMode() === "off") return;
  try {
    const answer = await api("sessions/" + session + "/chance");
    if (mine === chanceAsked) chance = answer;
  } catch (error) {
    if (mine === chanceAsked) chance = { error: error.message };
  }
  if (mine === chanceAsked) drawChance();
}

function drawChance() {
  // Only the line is read out to screen readers, not the guess buttons.
  const line = document.getElementById("chance-line");
  const more = document.getElementById("chance-more");
  line.textContent = "";
  line.classList.remove("error");
  more.replaceChildren();
  for (const chip of document.querySelectorAll(".seat-chance")) chip.remove();
  const mode = chanceMode();
  if (!state || !state.training || !state.your_turn || mode === "off") return;
  if (mode === "guess" && !revealed) {
    const guesses = element("div", "guesses");
    for (let tenth = 1; tenth <= 9; tenth++) {
      const chip = button(percent(tenth / 10), () => reveal(tenth / 10), "chip", String(tenth));
      chip.title = "About " + percent(tenth / 10) + " · Key: " + tenth;
      guesses.append(chip);
    }
    const skip = button("Show", () => reveal(null), "chip quiet", "s");
    skip.title = "See it without guessing -- key: S";
    guesses.append(skip);
    line.textContent = "Guess your chance to win:";
    more.append(guesses);
    return;
  }
  if (chance === null) {
    line.textContent = "Working it out...";
    return;
  }
  if (chance.error) {
    line.textContent = chance.error;
    line.classList.add("error");
    return;
  }
  const names = chance.opponents.map((opponent) => state.seats[opponent.seat].name);
  const against = names.length === 1 ? names[0] : names.length === 2 ? "both " + names.join(" and ") : names.length + " opponents";
  line.textContent = "Chance to win " + percent(chance.chance) + " against " + against;
  if (guess !== null) line.textContent += " -- you guessed " + percent(guess) + " (" + guessWords(guess, chance.chance) + ")";
  if (chance.opponents.some((opponent) => !opponent.acted)) {
    more.append(element("p", "hint-line", "Players still to act count as holding any hand; most of them will fold."));
  }
  chance.opponents.forEach((opponent, index) => seatChance(opponent, names[index]));
}

function reveal(choice) {
  guess = choice;
  revealed = true;
  drawChance();
}

function guessWords(guessed, actual) {
  const off = Math.round(100 * guessed) - Math.round(100 * actual);
  if (Math.abs(off) <= 5) return "close";
  return (Math.abs(off) <= 15 ? "a bit " : "too ") + (off > 0 ? "high" : "low");
}

function seatChance(opponent, name) {
  const text = "Your chance " + percent(opponent.chance);
  let chip;
  if (!opponent.acted) {
    chip = element("span", "seat-chance muted", text);
    chip.title = name + " hasn't acted yet, so this counts any two cards";
  } else if (opponent.width > 0.6) {
    // (where 1.0 is every hand) the grid would be unhelpful
    chip = element("span", "seat-chance muted", text);
    chip.title = name + " could still hold almost any hand";
  } else {
    chip = button(text, () => openRange(opponent, name), "seat-chance");
    chip.title = "Your chance to win against " + name + " alone, who likely holds about " +
      percent(opponent.width) + " of hands. Click to see them.";
  }
  document.querySelector(".seat[data-seat='" + opponent.seat + "']").append(chip);
}

function openRange(opponent, name) {
  const dialog = document.getElementById("range-dialog");
  const box = document.getElementById("range-box");
  const width = percent(opponent.width);
  box.replaceChildren(
    element("h3", "", name + "'s likely hands"),
    element("p", "", "About " + width + " of all hands, given how " + name + " has played. Brighter means more likely."),
  );
  if (state.board.length) {
    const board = element("p", "", "Board: ");
    for (const text of state.board) board.append(card(text, true));
    box.append(board);
  }
  box.append(
    rangeGrid(opponent.classes, name + " likely holds about " + width + " of hands"),
    element("p", "", "Your chance against " + name + " alone: " + percent(opponent.chance)),
    button("Close", () => dialog.close(), "secondary"),
  );
  dialog.showModal();
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
    startClock(actions);
    if (legal.fold && !legal.check) actions.append(button("Fold", () => act({ kind: "fold" }), "neutral", "f"));
    if (legal.check) actions.append(button("Check", () => act({ kind: "check" }), "call", "k"));
    if (legal.call !== null) {
      const allIn = legal.call >= you().stack ? " (all-in)" : "";
      actions.append(button("Call " + chips(legal.call) + allIn, () => act({ kind: "call" }), "call", "c"));
    }
    let raiseButton = null;
    if (legal.raise) {
      raiseButton = button("", () => act({ kind: legal.raise.kind, amount: Number(document.getElementById("amount").value) }), "raise", "r");
      actions.append(raiseButton);
    }
    if (state.coach) actions.append(button("Ask the coach", hint, "secondary", "h"));
    renderSizing(legal.raise, raiseButton);
  } else {
    clock = null;
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

function startClock(actions) {
  // The move timer: a pill first in the actions, counting down from when the turn began.
  if (!state.timer) return;
  if (!clock || clock.turn !== state.turn) {
    clock = { turn: state.turn, deadline: Date.now() + state.timer * 1000, warned: false, fired: false };
    document.getElementById("clock-live").textContent = "";
  }
  const pill = element("span", "clock");
  pill.id = "clock";
  actions.append(pill);
  tickClock();
}

function secondsLeft() {
  return Math.max(0, Math.ceil((clock.deadline - (clockPaused ?? Date.now())) / 1000));
}

function tickClock() {
  renderToTable();
  const pill = document.getElementById("clock");
  if (!clock || !pill) return;
  const left = secondsLeft();
  pill.textContent = left + " s left" + (clockPaused ? ", paused" : "");
  const low = left <= Math.min(10, state.timer / 3);
  pill.classList.toggle("low", low);
  if (low && !clock.warned) {
    clock.warned = true;
    document.getElementById("clock-live").textContent = left + " seconds left to act";
  }
  if (left === 0 && !clockPaused && !clock.fired) timeUp(clock.turn, state.legal.check);
}

async function timeUp(turn, checks) {
  // Check if that's free, otherwise fold; the server picks, and refuses a turn already over.
  if (busy) return;  // another request is on its way: the next tick tries again
  clock.fired = true;
  await run("table-error", async (current) => {
    let answer;
    try {
      answer = await api("sessions/" + session + "/action", { kind: "timeout", turn });
    } catch (error) {
      return;  // the player acted at the same moment
    }
    if (!current()) return;
    state = answer;
    timeoutNote = "Time ran out, so you " + (checks ? "checked" : "folded") + ".";
    clearCoach();
    render();
  });
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

function rowKey(row) {
  return row.session + "/" + row.hand;
}

function amount(value, row) {
  // A row's chips, or the same in that hand's big blinds (tournament blinds grow).
  return inBB ? value / row.big_blind : value;
}

function handsTable(rows, csv) {
  if (!rows.length) return element("p", "", "No finished hands yet.");
  const box = element("div", "hands-box");
  const redraw = (focus) => {
    const fresh = handsTable(rows.map((row) => latestRows.get(rowKey(row))), csv);
    box.replaceWith(fresh);
    fresh.querySelector(focus).focus();
  };
  const unit = makeSwitch("unit", "Big blinds", "Show Put in and Result in big blinds rather than chips", inBB, (input) => {
    inBB = input.checked;
    redraw("input[name=unit]");
  });
  const link = element("a", "download", "CSV");
  link.href = csv;
  link.title = "Download this session's hands as a CSV file";
  const tools = element("div", "table-tools");
  tools.append(unit, link);
  const scroll = element("div", "table-scroll");
  const table = element("table", "hands");
  const head = element("tr");
  for (const [index, [title, tip, className, value]] of COLUMNS.entries()) {
    const th = element("th", className);
    if (tip) th.title = tip;
    head.append(th);
    if (!value) {
      th.textContent = title;
      continue;
    }
    // Each click on a title goes up, down, then back to the newest hand first.
    const order = handsOrder && handsOrder.column === index ? handsOrder : null;
    th.setAttribute("aria-sort", order ? (order.up ? "ascending" : "descending") : "none");
    th.append(button(title + (order ? (order.up ? " ▲" : " ▼") : ""), () => {
      handsOrder = !order ? { column: index, up: true } : order.up ? { column: index, up: false } : null;
      redraw("th:nth-child(" + (index + 1) + ") button");
    }, "sort"));
  }
  table.append(head);
  const shown = rows.slice().reverse();
  if (handsOrder) {
    const value = COLUMNS[handsOrder.column][3];
    const sign = handsOrder.up ? 1 : -1;
    // Blanks (a rating still coming, or none) stay last either way.
    shown.sort((a, b) => (value(a) === "") - (value(b) === "") || sign * (value(a) - value(b)));
  }
  for (const row of shown) {
    const key = rowKey(row);
    const tr = element("tr", "hand-row");
    tr.dataset.hand = key;
    latestRows.set(key, row);
    const toggle = button(String(row.hand), () => {
      if (openMoves.delete(key)) {
        tr.nextElementSibling.remove();
      } else {
        openMoves.add(key);
        tr.after(movesRow(latestRows.get(key)));
      }
      toggle.setAttribute("aria-expanded", String(openMoves.has(key)));
    }, "moves-toggle");
    toggle.title = "Show or hide the moves and the review";
    toggle.setAttribute("aria-expanded", String(openMoves.has(key)));
    tr.addEventListener("click", (event) => {
      if (!event.target.closest("button")) toggle.click();
    });
    const number = element("td");
    number.append(toggle);
    const cards = element("td");
    for (const text of row.cards.split(" ").filter(Boolean)) cards.append(card(text, true));
    const board = element("td", "wide-only");
    for (const text of row.board.split(" ").filter(Boolean)) board.append(card(text, true));
    const chance = element("td", "number chance");
    const rating = element("td", "number rating");
    fillRated(chance, rating, row);
    const result = element("td", "number" + (row.result > 0 ? " up" : row.result < 0 ? " down" : ""), signed(amount(row.result, row)));
    tr.append(number, element("td", "", POSITIONS[row.position] || "early"), cards, element("td", "number strength", row.strength), board);
    tr.append(chance, element("td", "wide-only number", chips(amount(row.put_in, row))), result, rating);
    table.append(tr);
    if (openMoves.has(key)) table.append(movesRow(row));
  }
  scroll.append(table);
  box.append(tools, scroll);
  return box;
}

function fillRated(chance, rating, row) {
  // Empty while the hand is still being rated, a muted dash when there is nothing to show.
  for (const cell of [chance, rating]) {
    cell.replaceChildren();
    cell.classList.remove("muted");
    cell.removeAttribute("title");
  }
  if (row.pending) return;
  if (row.rating === "") {
    for (const cell of [chance, rating]) {
      cell.textContent = "–";
      cell.classList.add("muted");
      cell.title = row.acted ? "Not rated" : "You made no move this hand";
    }
    return;
  }
  chance.textContent = row.chance;
  rating.textContent = row.rating.toFixed(2);
  rating.title = row.moves;
}

function movesRow(row) {
  // A loaded review moves into the redrawn table as it is, keeping its opened details.
  const key = rowKey(row);
  const kept = reviews.has(key) && document.querySelector(`.row-review[data-hand="${CSS.escape(key)}"]`);
  const tr = element("tr", "moves");
  const cell = element("td");
  cell.colSpan = COLUMNS.length;
  cell.append(kept || rowReview(row));
  tr.append(cell);
  return tr;
}

function rowReview(row, failed = false) {
  // The hand's full review once it is rated, its moves and "Loading…" until then. A hand whose
  // review couldn't be fetched (an unsaved session) shows its moves and rated summary instead.
  const key = rowKey(row);
  const box = element("div", "row-review copy-host");
  box.dataset.hand = key;
  const answer = reviews.get(key);
  if (answer) {
    const tools = element("div", "review-tools");
    tools.append(copyIcon("copy", copyTip(row.hand), () => answer.texts[copyMode()]));
    box.append(tools, coachView(answer, true));
    return box;
  }
  if (failed && row.review !== null) {
    box.append(handSummary(row.review.at(-1), row.rating === "" ? null : row.rating, row.band));
  }
  for (const line of row.history.split("\n")) box.append(storyLine(line));
  if (failed && row.review !== null) {
    for (const line of row.review.slice(0, -1)) box.append(storyLine(line));
    return box;
  }
  box.append(element("p", "waiting", "Loading…"));
  if (!row.pending) loadReview(row);
  return box;
}

async function loadReview(row) {
  const key = rowKey(row);
  if (reviews.has(key) || loading.has(key)) return;
  loading.add(key);
  let failed = false;
  try {
    reviews.set(key, await api("history/" + encodeURIComponent(row.session) + "/review", { hand: row.hand }));
  } catch (error) {
    failed = true;  // not kept, so the next open tries again
  }
  loading.delete(key);
  document.querySelector(`.row-review[data-hand="${CSS.escape(key)}"]`)?.replaceWith(rowReview(latestRows.get(key), failed));
}

function handSummary(text, rating, band, result) {
  const box = element("div", "hand-summary");
  if (rating !== null) {
    const badge = element("span", "rating-badge " + band, "Hand rating " + rating.toFixed(2));
    badge.title = RATING_TIP;
    box.append(badge);
  }
  box.append(prettyLine(text));
  if (result) box.append(element("p", "result", result));
  return box;
}

function refreshRows(path, container, pending) {
  // While hands are still being rated, re-read them once a second, until none are left or the
  // tables show something else.
  clearTimeout(refreshTimer);
  const asked = ++refreshes;
  if (!pending) return;
  refreshTimer = setTimeout(async () => {
    let answer;
    try {
      answer = await api(path);
    } catch (error) {
      return;
    }
    if (asked !== refreshes) return;
    // Only the rated cells and the opened rows still loading change, so nothing else moves.
    for (const row of answer.hands || answer.rows) {
      const key = rowKey(row);
      latestRows.set(key, row);
      const tr = container.querySelector(`tr[data-hand="${CSS.escape(key)}"]`);
      if (!tr) continue;
      fillRated(tr.querySelector("td.chance"), tr.querySelector("td.rating"), row);
      const opened = tr.nextElementSibling;
      if (opened && opened.classList.contains("moves") && !reviews.has(key)) opened.querySelector(".row-review").replaceWith(rowReview(row));
    }
    refreshRows(path, container, answer.pending);
  }, 1000);
}

function renderHands() {
  document.getElementById("net").textContent = state.session_net ? "· " + signed(state.session_net) : "";
  document.getElementById("hands-table").replaceChildren(handsTable(state.hands, "/api/sessions/" + session + "/hands.csv"));
  refreshRows("sessions/" + session, document.getElementById("hands-table"), state.pending);
}

function render() {
  renderTable();
  renderStatus();
  renderActions();
  renderLog();
  renderSwitches();
  renderHands();
  showChance();
  renderToTable();
}

function clearCoach() {
  coachShown = null;
  document.getElementById("coach").replaceChildren();
  document.getElementById("coach-panel").hidden = true;
}

async function play(command, body, fresh = true) {
  // A move, and the table it leads to; `fresh` clears the coach and the timer's note.
  await run("table-error", async (current) => {
    const answer = await api("sessions/" + session + "/" + command, body);
    if (!current()) return;
    state = answer;
    if (fresh) {
      timeoutNote = "";
      clearCoach();
    }
    render();
  });
}

function act(action) {
  return play("action", action);
}

function next() {
  return play("next", {});
}

function rebuy() {
  return play("rebuy", {}, false);
}


function newGame() {
  if (!window.confirm("Leave this game and start a new one?")) return;
  session = null;
  state = null;
  clock = null;
  timeoutNote = "";
  clearCoach();
  document.getElementById("table-error").textContent = "";
  show("setup");
}

async function coach(path, waiting) {
  await run("table-error", async (current) => {
    const panel = document.getElementById("coach-panel");
    const target = document.getElementById("coach");
    panel.hidden = false;
    coachShown = null;
    target.replaceChildren(element("p", "waiting", waiting));
    if (clock) clockPaused = Date.now();
    tickClock();
    try {
      const answer = await api("sessions/" + session + "/" + path, {});
      if (!current()) return;
      coachShown = { answer, full: path === "review" };
      target.replaceChildren(coachView(answer, coachShown.full));
    } catch (error) {
      target.replaceChildren();
      panel.hidden = true;
      throw error;
    } finally {
      if (clock && clockPaused !== null) clock.deadline += Date.now() - clockPaused;
      clockPaused = null;
      tickClock();
    }
  });
}

function hint() {
  return coach("hint", "Thinking...");
}

function analyze() {
  return coach("analyze", "Checking your last move...");
}

function review() {
  return coach("review", "Reviewing the hand...");
}

// The coach's answers


function optionsTable(decision, god) {
  // What each option averages: against a strong player, and with God's view against these
  // players' styles and, in hindsight, against the cards they held.
  const table = element("table", "options");
  const head = element("tr");
  const titles = [
    ["Option", ""],
    ["Strong player", "How often a strong player makes this move here"],
    ["Vs strong", "Average result against a strong player"],
  ];
  if (god) {
    titles.push(
      ["Vs styles", "Average result against the hands these players' styles would hold, played their way"],
      ["Vs cards", "In hindsight: the average result if you could see their cards, an estimate before the river"],
    );
  }
  for (const [index, [title, tip]] of titles.entries()) {
    const cell = element("th", index > 1 ? "number" : "", title);
    if (tip) cell.title = tip;
    head.append(cell);
  }
  table.append(head);
  const value = (worth, noise) => {
    if (worth === null) return element("td", "number muted", "–");
    const cell = element("td", "number", decision.unit === "chips" ? signed(worth) : worth.toFixed(1) + "%");
    if (noise) cell.title = "± " + chips(noise) + ": an estimate from sampling";
    return cell;
  };
  for (const option of decision.options) {
    const tr = element("tr", option.best ? "best" : "");
    const name = element("td");
    name.append(element("span", "action", option.action));
    if (option.chosen) name.append(element("span", "mark", "your move"));
    if (option.best) {
      const mark = element("span", "mark best", "best");
      mark.title = "best against a strong player";
      name.append(mark);
    }
    const share = element("td", "share");
    const track = element("span", "track");
    const bar = element("span", "bar");
    bar.style.width = Math.round(100 * option.strong_share) + "%";
    track.append(bar);
    share.append(track, element("span", "", percent(option.strong_share)));
    tr.append(name, share, value(option.vs_strong, option.vs_strong_noise));
    if (god) tr.append(value(option.vs_bots, option.vs_bots_noise), value(option.vs_cards, option.vs_cards_noise));
    table.append(tr);
  }
  const scroll = element("div", "options-scroll");
  scroll.append(table);
  return scroll;
}

function rangeGrid(classes, label) {
  const grid = element("div", "grid");
  grid.setAttribute("role", "img");
  grid.setAttribute("aria-label", label);
  for (let row = 0; row < 13; row++) {
    for (let column = 0; column < 13; column++) {
      const a = RANKS[Math.min(row, column)];
      const b = RANKS[Math.max(row, column)];
      const name = row === column ? a + b : a + b + (row < column ? "s" : "o");
      const weight = classes[row * 13 + column];
      const cell = element("span", weight > 0.55 ? "cell strong" : "cell", name);
      cell.style.setProperty("--w", String(weight));
      cell.title = name + ": " + percent(weight);
      grid.append(cell);
    }
  }
  return grid;
}

function decisionView(decision, open, god) {
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
  const facts = decision.summary.slice();
  if (decision.chosen) box.append(element("p", "note", facts.shift()));
  box.append(element("p", "", facts.join(" ")));
  if (god && decision.god_summary.length) {
    const hindsight = element("div", "hindsight");
    hindsight.append(element("p", "hindsight-title", "God's view"), element("p", "", decision.god_summary.join(" ")));
    box.append(hindsight);
  }
  box.append(element("p", "note", "Results in " + decision.unit), optionsTable(decision, god));
  const details = element("details");
  details.open = open;
  details.append(element("summary", "", "Show details"));
  const list = element("ul");
  for (const line of decision.details.concat(god ? decision.god_details : [])) list.append(element("li", "", line));
  details.append(list);
  for (const range of god ? decision.god_ranges : decision.ranges) {
    if (range.width >= 0.6) continue;  // nearly every hand: the grid would be solid colour
    const label = range.name + " likely holds about " + percent(range.width) + " of hands";
    details.append(element("p", "hint-line", range.name + "'s likely hands (brighter means more likely):"), rangeGrid(range.classes, label));
  }
  box.append(details);
  return box;
}

function coachView(answer, full) {
  // A hand review follows the switches, its rating first; a hint or a move check always shows
  // the coach's numbers from what the player could know.
  const solver = !full || switches.solver;
  const god = full && switches.god && answer.finished;
  const view = element("div", "coach-view");
  if (answer.summary) view.append(handSummary(answer.summary, answer.rating, answer.band, answer.result));
  const story = element("details", "story");
  story.open = full;
  story.append(element("summary", "", full ? "The hand" : "The hand so far"));
  for (const line of god ? answer.god_history : answer.history) story.append(storyLine(line));
  view.append(story);
  if (solver) {
    if (!answer.decisions.length) view.append(element("p", "", "You made no decision in that hand."));
    for (const decision of answer.decisions) view.append(decisionView(decision, false, god));
  } else {
    for (const line of answer.rated) view.append(element("p", line.startsWith(" ") ? "rated-move" : "", line.trim()));
  }
  if (answer.result && !answer.summary) view.append(element("p", "result", answer.result));
  return view;
}

function switchChanged(input) {
  // One state for the page: every copy of the switch follows, and each review shown is redrawn.
  switches[input.name] = input.checked;
  for (const other of document.querySelectorAll(".review-switches input[name=" + input.name + "]")) other.checked = input.checked;
  if (session && state) renderSwitches();
  if (coachShown && coachShown.full) document.getElementById("coach").replaceChildren(coachView(coachShown.answer, true));
  for (const box of document.querySelectorAll(".row-review")) box.replaceWith(rowReview(latestRows.get(box.dataset.hand)));
}

// Past sessions

function gameName(saved) {
  if (saved.mode === "training") return trainingText(saved.training, saved.players);
  return saved.mode === "cash" ? "Cash game" : "Tournament";
}

async function openHistory() {
  show("history");
  closeSession();
  await run("history-error", async () => listSessions(await api("history")));
}

function listSessions(answer) {
  const target = document.getElementById("sessions");
  if (!answer.logging) {
    target.replaceChildren(element("p", "", "Sessions aren't saved: the table was started with --no-log."));
    return;
  }
  if (!answer.sessions.length) {
    target.replaceChildren(element("p", "", "No saved sessions yet. Play a game and it will show up here."));
    return;
  }
  const picks = [];
  const remove = button("Delete selected", () => deleteSessions(picks.filter((pick) => pick.checked).map((pick) => pick.value)), "danger");
  remove.disabled = true;
  const all = element("input");
  all.type = "checkbox";
  all.setAttribute("aria-label", "Select all sessions");
  const update = () => {
    const count = picks.filter((pick) => pick.checked).length;
    remove.disabled = !count;
    all.checked = count === picks.length;
    all.indeterminate = count > 0 && count < picks.length;
  };
  all.addEventListener("change", () => {
    for (const pick of picks) pick.checked = all.checked;
    update();
  });
  const table = element("table", "hands");
  const head = element("tr");
  const dateHead = element("th");
  const allLabel = element("label", "inline");
  allLabel.append(all, "Date");
  dateHead.append(allLabel);
  head.append(dateHead);
  for (const [title, className] of [["Game"], ["Players", "wide-only"], ["Opponents", "wide-only"], ["Hands", "number"], ["Result", "number"], ["", "wide-only"]]) head.append(element("th", className, title));
  table.append(head);
  for (const saved of answer.sessions) {
    const tr = element("tr", "session-row");
    tr.dataset.name = saved.name;
    tr.title = "Show or hide this session's hands"
    tr.addEventListener("click", (event) => {
      if (!event.target.closest("button, label")) openSession(saved);
    });
    const pick = element("input");
    pick.type = "checkbox";
    pick.value = saved.name;
    pick.setAttribute("aria-label", "Select " + gameName(saved) + " on " + saved.date);
    pick.addEventListener("change", update);
    picks.push(pick);
    const date = element("td");
    const label = element("label", "inline");
    // "Oct 9", with the year only when it isn't this year; the full date shows on hover.
    const day = new Date(saved.date + "T00:00");
    const year = day.getFullYear() === new Date().getFullYear() ? undefined : "numeric";
    label.append(pick, day.toLocaleDateString("en-US", { month: "short", day: "numeric", year }));
    label.title = saved.date;
    date.append(label);
    tr.append(date, element("td", "", gameName(saved)));
    tr.append(element("td", "wide-only", saved.players), element("td", "wide-only", DIFFICULTY[saved.difficulty]), element("td", "number", saved.hands));
    const training = saved.mode === "training";
    const result = element("td", "number" + (training ? " muted" : saved.net > 0 ? " up" : saved.net < 0 ? " down" : ""), signed(saved.net));
    if (training) result.title = "Training: not counted in your results";
    tr.append(result);
    const open = element("td", "wide-only");
    open.append(button("Open", () => openSession(saved), "small session-toggle"));
    tr.append(open);
    table.append(tr);
  }
  const actions = element("p", "session-tools");
  actions.append(remove);
  const scroll = element("div", "table-scroll");
  scroll.append(table);
  target.replaceChildren(actions, scroll);
  markSession();
}

function markSession() {
  // The open session's row reads Close, the others Open.
  const open = document.getElementById("session-hands").dataset.name;
  for (const tr of document.querySelectorAll("#sessions tr.session-row")) {
    const toggle = tr.querySelector(".session-toggle");
    toggle.textContent = tr.dataset.name === open ? "Close" : "Open";
    toggle.setAttribute("aria-expanded", String(tr.dataset.name === open));
  }
}

function closeSession() {
  const target = document.getElementById("session-hands");
  target.replaceChildren();
  delete target.dataset.name;
  refreshRows("", target, false);
  markSession();
}

async function openSession(saved) {
  // Opening the open session again closes it.
  const target = document.getElementById("session-hands");
  if (target.dataset.name === saved.name) {
    closeSession();
    return;
  }
  const name = encodeURIComponent(saved.name);
  await run("history-error", async () => {
    const answer = await api("history/" + name);

    target.dataset.name = saved.name;
    const title = element("h2", "", "Hands: " + gameName(saved) + " on " + saved.date);
    target.replaceChildren(title, reviewSwitches(), handsTable(answer.rows, "/api/history/" + name + "/hands.csv"));
    refreshRows("history/" + name, target, answer.pending);
    markSession();
  });
}

async function deleteSessions(names) {
  await run("history-error", async () => {
    const answer = await api("history/delete", { names });
    for (const key of [...openMoves]) if (names.some((name) => key.startsWith(name + "/"))) openMoves.delete(key);
    for (const key of [...reviews.keys()]) if (names.some((name) => key.startsWith(name + "/"))) reviews.delete(key);
    if (names.includes(document.getElementById("session-hands").dataset.name)) closeSession();
    listSessions(answer);
  });
}

// Setup and page-wide controls

const setupForm = document.getElementById("setup-form");
setupForm.querySelector("details.more").before(makeSwitch("coach", "Coach", "Hints and checks of your moves while you play", true, null));
let preview = 0;

function selectedMode() {
  return document.querySelector("input[name=mode]:checked").value;
}

function handsSpec() {
  // The training hands as the server reads them: a preset, "low-high" for a custom window, or
  // "" for any hand.
  const preset = document.getElementById("hands-preset").value;
  if (preset !== "custom") return preset;
  const low = document.getElementById("hands-low").value || "0";
  const high = document.getElementById("hands-high").value || "100";
  return low + "-" + high;
}

function fillSeats() {
  const select = document.getElementById("position");
  const players = Number(setupForm.elements.seats.value);
  if (!(players >= 2 && players <= 8)) return;  // mid-typing: keep the list and the choice
  const kept = select.value;
  const options = [["", "Change seat every hand"]];
  for (const code of seatNames(players)) options.push([code, seatWords(code, players)]);
  select.replaceChildren(...options.map(([value, text]) => {
    const option = element("option", "", text);
    option.value = value;
    return option;
  }));
  select.value = options.some(([value]) => value === kept) ? kept : "";
}

async function showHands() {
  // The preview grid for the chosen hands, from the server's own rules; a slower answer that
  // arrives after a newer choice is dropped.
  const preset = document.getElementById("hands-preset").value;
  // The two numbers belong to a window of the strength order, not to "any hand" or pairs.
  document.querySelector(".range-fields").hidden = preset !== "custom" && !/^\d/.test(preset);
  const target = document.getElementById("hands-preview");
  const spec = handsSpec();
  const mine = ++preview;
  if (!spec) {
    target.replaceChildren(element("p", "hint-line", "Any two cards, as in a normal game."));
    return;
  }
  try {
    const answer = await api("hands/" + encodeURIComponent(spec));
    if (mine !== preview) return;
    const words = answer.count + " hand types, about " + percent(answer.share) + " of deals";
    target.replaceChildren(element("p", "hint-line", words), rangeGrid(answer.classes, "Hands you will be dealt: " + words));
  } catch (error) {
    if (mine === preview) target.replaceChildren(element("p", "error", error.message));
  }
}

function presetChanged() {
  const preset = document.getElementById("hands-preset").value;
  const bounds = /^(\d+)-(\d+)$/.exec(preset);
  if (bounds) {
    document.getElementById("hands-low").value = bounds[1];
    document.getElementById("hands-high").value = bounds[2];
  }
  showHands();
}

function rangeTyped() {
  document.getElementById("hands-preset").value = "custom";
  showHands();
}

function tierHint() {
  // With picked training cards the server keeps Expert's read of you neutral.
  const tier = setupForm.elements.tier.value;
  const picked = tier === "4" && selectedMode() === "training" && handsSpec() !== "";
  document.getElementById("tier-hint").textContent = DIFFICULTY_WORDS[tier] + (picked ? " In training with picked cards it plays like Hard." : "");
}

function modeChanged() {
  const mode = selectedMode();
  document.getElementById("blinds-field").hidden = mode === "tournament";
  document.getElementById("training-options").hidden = mode !== "training";
  setupForm.elements.coach.checked = mode !== "tournament";  // DESIGN 7.8: off in tournaments
  if (mode === "training") showHands();
}

for (const radio of document.querySelectorAll("input[name=mode]")) radio.addEventListener("change", modeChanged);
setupForm.elements.seats.addEventListener("input", fillSeats);
document.getElementById("hands-preset").addEventListener("change", presetChanged);
for (const id of ["hands-low", "hands-high"]) document.getElementById(id).addEventListener("input", rangeTyped);
for (const id of ["hands-preset", "hands-low", "hands-high"]) document.getElementById(id).addEventListener("input", tierHint);
for (const radio of document.querySelectorAll("input[name=mode]")) radio.addEventListener("change", tierHint);
setupForm.elements.tier.addEventListener("change", tierHint);
fillSeats();
tierHint();

setupForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  const options = {
    mode: form.get("mode"),
    seats: Number(form.get("seats")),
    tier: Number(form.get("tier")),
    coach: form.get("coach") === "on",
  };
  if (options.mode !== "tournament") options.blinds = form.get("blinds");
  if (form.get("stack")) options.stack = Number(form.get("stack"));
  if (form.get("timer")) options.timer = Number(form.get("timer"));
  if (options.mode === "training") {
    if (form.get("position")) options.position = form.get("position");
    if (handsSpec()) options.hands = handsSpec();
  }
  run("setup-error", async () => {
    const created = await api("sessions", options);
    session = created.id;
    state = created.state;
    clock = null;
    timeoutNote = "";
    clearCoach();
    show("table");
    render();
  });
});

document.getElementById("new-game").addEventListener("click", newGame);
document.getElementById("open-history").addEventListener("click", openHistory);
document.getElementById("history-back").addEventListener("click", () => show("setup"));
document.getElementById("words-link").addEventListener("click", () => { document.getElementById("glossary").open = true; });

const helpersHidden = window.localStorage.getItem(HELPERS_KEY) === "1";
document.body.classList.toggle("no-helpers", helpersHidden);
const helpers = makeSwitch("hide-helpers", "Hide helpers", HELPERS_TIP, helpersHidden, (input) => {
  window.localStorage.setItem(HELPERS_KEY, input.checked ? "1" : "0");
  document.body.classList.toggle("no-helpers", input.checked);
});
helpers.id = "helpers-toggle";
helpers.hidden = true;
document.getElementById("words-link").before(helpers);

const chanceSaved = window.localStorage.getItem(CHANCE_KEY);
for (const radio of document.querySelectorAll("input[name=chance]")) {
  radio.checked = radio.value === (["show", "guess"].includes(chanceSaved) ? chanceSaved : "off");
  radio.addEventListener("change", () => {
    window.localStorage.setItem(CHANCE_KEY, radio.value);
    revealed = false;
    guess = null;
    if (chance === null) showChance();
    else drawChance();
  });
}

document.querySelector("#table .copy-host").prepend(reviewSwitches());
setInterval(tickClock, 250);


new IntersectionObserver(([entry]) => {
  controlsAbove = !entry.isIntersecting && entry.boundingClientRect.bottom < 0;
  renderToTable();
}).observe(document.getElementById("controls"));
document.getElementById("to-table").addEventListener("click", (event) => {
  const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  window.scrollTo({ top: 0, behavior: smooth ? "smooth" : "auto" }); // the top seat sits above the table box
  event.currentTarget.blur();
});

const rangeDialog = document.getElementById("range-dialog");
rangeDialog.addEventListener("click", (event) => {
  if (event.target === rangeDialog) rangeDialog.close();  // a click on the backdrop
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
