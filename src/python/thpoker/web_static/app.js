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
const DIFFICULTY = { 1: "Easy", 2: "Medium", 3: "Hard" };
const STREETS = { preflop: "Before the flop", flop: "Flop", turn: "Turn", river: "River" };
const PRESETS = [["⅓ pot", 1 / 3], ["½ pot", 1 / 2], ["⅔ pot", 2 / 3], ["Pot", 1]];
const CARD = /^[2-9TJQKA][cdhs]$/;
const HELPERS_KEY = "thpoker.hideHelpers";
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
const COPY_TIP = "Copy the hand as you saw it, with your moves rated 0 to 1, for an AI chat";
const ANALYSIS_TIP = "Copy the hand with the coach's full analysis and ratings, for an AI chat";
const COLUMNS = [
  ["Hand"], ["Seat"], ["Your cards"],
  ["Strength", "How your two starting cards rank among all starting hands: best 6% means only 6% are as strong or stronger"],
  ["Board", null, "wide-only"],
  ["Chance to win", "Your chance to win at showdown at your last move, reading the others' hands from their play the way a strong player would, so it can differ from the chance shown during play"],
  ["Put in", null, "wide-only"], ["Result"],
  ["Rating", "Your moves this hand, rated 0 to 1, with moves in bigger pots counting for more: 1 means you picked the best option every time"],
  [""],
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
    if (node.classList.contains("waiting")) return;
    const panel = node.closest(".panel");
    const pending = Promise.resolve(typeof text === "function" ? text() : text);
    node.classList.add("waiting");
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
      node.classList.remove("waiting");
    }
  });
  return node;
}

function tools(id) {
  // A panel's copy icons, emptied along with any copy box they left behind.
  const target = document.getElementById(id);
  target.closest(".panel").querySelector(".copy-fallback")?.remove();
  target.replaceChildren();
  return target;
}

function copyTools(id, answer) {
  tools(id).append(copyIcon("copy", COPY_TIP, answer.hand_text), copyIcon("chat", ANALYSIS_TIP, answer.copy_text));
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
  refreshRows(null, null, 0);
  for (const id of ["setup", "table", "history"]) document.getElementById(id).hidden = id !== view;
  document.getElementById("new-game").hidden = view !== "table";
  document.getElementById("helpers-toggle").hidden = view !== "table";
  document.getElementById("training-pill").hidden = view !== "table" || !state || !state.training;
  document.getElementById("chance-mode").hidden = view !== "table" || !state || !state.training;
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
  document.getElementById("status").textContent = statusText();
  document.body.classList.toggle("your-turn", Boolean(state.your_turn));
  const odds = document.getElementById("odds");
  odds.textContent = "";
  odds.title = "";
  if (state.your_turn && state.call_needs) {
    odds.textContent = "Calling needs " + percent(state.call_needs) + " to win";
    odds.title = "Pot odds: call " + chips(state.legal.call) + " into a pot of " + chips(state.pot)
      + ", so calling pays if you win more than " + percent(state.call_needs) + " of the time.";
  }
  const pill = document.getElementById("training-pill");
  pill.hidden = !state.training;
  if (state.training) pill.textContent = trainingText(state.training, state.seats.length);
  document.getElementById("chance-mode").hidden = !state.training;
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
      const chip = button(tenth * 10 + "%", () => reveal(tenth / 10), "chip", String(tenth));
      chip.title = "Guess " + tenth * 10 + "% -- key: " + tenth;
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
    more.append(element("p", "still", "Players still to act count as holding any two cards; most of them will fold."));
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
  const logTools = tools("log-tools");
  if (state.hand_number) {
    const path = "sessions/" + session + "/hands/" + state.hand_number + "/copy";
    logTools.append(copyIcon("copy", COPY_TIP, () => api(path).then((answer) => answer.text)));
  }
}

function handsTable(rows, reviewer) {
  if (!rows.length) return element("p", "", "No finished hands yet.");
  const table = element("table", "hands");
  const head = element("tr");
  for (const [title, tip, className] of COLUMNS) {
    const th = element("th", className, title);
    if (tip) th.title = tip;
    head.append(th);
  }
  table.append(head);
  for (const row of rows.slice().reverse()) {
    const tr = element("tr");
    const key = row.session + "/" + row.hand;
    tr.dataset.key = key;
    latestRows.set(key, row);
    const number = element("td");
    const toggle = button(String(row.hand), () => {
      if (openMoves.delete(key)) {
        tr.nextElementSibling.remove();
      } else {
        openMoves.add(key);
        tr.after(movesRow(latestRows.get(key)));
      }
      toggle.setAttribute("aria-expanded", String(openMoves.has(key)));
    }, "moves-toggle");
    toggle.title = "Show or hide the moves";
    toggle.setAttribute("aria-expanded", String(openMoves.has(key)));
    number.append(toggle);
    tr.append(number, element("td", "", POSITIONS[row.position] || "early"));
    const cards = element("td");
    for (const text of row.cards.split(" ").filter(Boolean)) cards.append(card(text, true));
    const board = element("td", "wide-only");
    for (const text of row.board.split(" ").filter(Boolean)) board.append(card(text, true));
    const chance = element("td", "number chance");
    const rating = element("td", "number rating");
    fillRated(chance, rating, row);
    tr.append(cards, element("td", "number strength", row.strength), board, chance, element("td", "wide-only", chips(row.put_in)));
    tr.append(element("td", row.result > 0 ? "up" : row.result < 0 ? "down" : "", signed(row.result)), rating);
    const action = element("td");
    if (reviewer) action.append(button("Review", () => reviewer(row.hand), "small"));
    tr.append(action);
    table.append(tr);
    if (openMoves.has(key)) table.append(movesRow(row));
  }
  const scroll = element("div", "table-scroll");
  scroll.append(table);
  return scroll;
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
  // The moves stay in view when a narrow screen scrolls the table sideways.
  const tr = element("tr", "moves");
  const cell = element("td");
  cell.colSpan = COLUMNS.length;
  const box = element("div", "moves-box");
  for (const line of row.history.split("\n")) box.append(prettyLine(line, line.startsWith(" ") ? "story-line" : "story-head"));
  box.append(rowReview(row));
  cell.append(box);
  tr.append(cell);
  return tr;
}

function rowReview(row) {
  const box = element("div", "row-review");
  if (row.review === null) {
    box.append(element("p", "waiting", "Loading…"));
  } else {
    for (const line of row.review) box.append(prettyLine(line.trim(), line.startsWith(" ") ? "story-line" : "story-head"));
  }
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
    // Only the rated cells and the opened rows change, so nothing else moves.
    for (const row of answer.hands || answer.rows) {
      const key = row.session + "/" + row.hand;
      latestRows.set(key, row);
      const tr = container.querySelector(`tr[data-key="${CSS.escape(key)}"]`);
      if (!tr) continue;
      fillRated(tr.querySelector("td.chance"), tr.querySelector("td.rating"), row);
      const opened = tr.nextElementSibling;
      if (opened && opened.classList.contains("moves")) opened.querySelector(".row-review").replaceWith(rowReview(row));
    }
    refreshRows(path, container, answer.pending);
  }, 1000);
}

function renderHands() {
  document.getElementById("net").textContent = state.session_net ? "· " + signed(state.session_net) : "";
  document.getElementById("hands-table").replaceChildren(handsTable(state.hands, null));
  document.getElementById("csv").href = "/api/sessions/" + session + "/hands.csv";
  refreshRows("sessions/" + session, document.getElementById("hands-table"), state.pending);
}

function render() {
  renderTable();
  renderStatus();
  renderActions();
  renderLog();
  renderHands();
  showChance();
}

function clearCoach() {
  document.getElementById("coach").replaceChildren();
  tools("coach-tools");
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
    tools("coach-tools");
    target.replaceChildren(element("p", "waiting", waiting));
    try {
      const answer = await api("sessions/" + session + "/" + path, {});
      if (!current()) return;
      target.replaceChildren(coachView(answer, path === "review"));
      copyTools("coach-tools", answer);
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
  return coach("analyze", "Checking your last move...");
}

function review() {
  return coach("review", "Reviewing the hand...");
}

// The coach's answers

function optionsTable(decision) {
  const table = element("table", "options");
  const head = element("tr");
  const unit = decision.unit === "chips" ? "" : " (" + decision.unit + ")";
  const titles = [
    ["Option", ""],
    ["Strong player", "How often a strong player makes this move here"],
    ["vs these players" + unit, "Average result against the players at this table"],
    ["vs a strong player" + unit, "Average result against a strong player"],
  ];
  for (const [title, tip] of titles) {
    const cell = element("th", "", title);
    if (tip) cell.title = tip;
    head.append(cell);
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
      const cell = element("span", "cell", name);
      cell.style.setProperty("--w", String(weight));
      cell.title = name + ": " + percent(weight);
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
    const label = range.name + " likely holds about " + percent(range.width) + " of hands";
    details.append(element("p", "grid-title", range.name + "'s likely hands (brighter means more likely):"), rangeGrid(range.classes, label));
  }
  box.append(details);
  return box;
}

function coachView(answer, full) {
  const view = element("div", "coach-view");
  const story = element("details", "story");
  story.open = full;
  story.append(element("summary", "", full ? "The hand" : "The hand so far"));
  for (const line of answer.history) story.append(prettyLine(line, line.startsWith(" ") ? "story-line" : "story-head"));
  view.append(story);
  if (!answer.decisions.length) view.append(element("p", "", "You made no decision in that hand."));
  for (const decision of answer.decisions) view.append(decisionView(decision, false));
  if (answer.summary) view.append(prettyLine(answer.summary, "hand-summary"));
  if (answer.result) view.append(element("p", "result", answer.result));
  return view;
}

// Past sessions

function gameName(saved) {
  if (saved.mode === "training") return trainingText(saved.training, saved.players);
  return saved.mode === "cash" ? "Cash game" : "Tournament";
}

async function openHistory() {
  show("history");
  document.getElementById("session-hands").replaceChildren();
  document.getElementById("history-coach-panel").hidden = true;
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
  dateHead.append(all, " Date");
  head.append(dateHead);
  for (const title of ["Game", "Players", "Opponents", "Hands", "Result", ""]) head.append(element("th", "", title));
  table.append(head);
  for (const saved of answer.sessions) {
    const tr = element("tr");
    const pick = element("input");
    pick.type = "checkbox";
    pick.value = saved.name;
    pick.setAttribute("aria-label", "Select " + gameName(saved) + " on " + saved.date);
    pick.addEventListener("change", update);
    picks.push(pick);
    const date = element("td");
    const label = element("label", "inline");
    label.append(pick, saved.date);
    date.append(label);
    tr.append(date, element("td", "", gameName(saved)));
    tr.append(element("td", "", saved.players), element("td", "", DIFFICULTY[saved.difficulty]), element("td", "", saved.hands));
    const training = saved.mode === "training";
    const result = element("td", training ? "muted" : saved.net > 0 ? "up" : saved.net < 0 ? "down" : "", signed(saved.net));
    if (training) result.title = "Training: not counted in your results";
    tr.append(result);
    const open = element("td");
    open.append(button("Open", () => openSession(saved), "small"));
    tr.append(open);
    table.append(tr);
  }
  const actions = element("p", "session-tools");
  actions.append(remove);
  const scroll = element("div", "table-scroll");
  scroll.append(table);
  target.replaceChildren(actions, scroll);
}

async function openSession(saved) {
  const name = saved.name;
  const label = gameName(saved) + " on " + saved.date;
  await run("history-error", async () => {
    const answer = await api("history/" + encodeURIComponent(name));
    const target = document.getElementById("session-hands");
    target.dataset.name = name;
    const link = element("a", "download", "CSV");
    link.href = "/api/history/" + encodeURIComponent(name) + "/hands.csv";
    link.title = "Download this session as a CSV file";
    link.prepend(icon("download"));
    const actions = element("p", "session-tools");
    actions.append(link);
    target.replaceChildren(element("h2", "", "Hands: " + label), actions, handsTable(answer.rows, (hand) => reviewSaved(name, hand)));
    refreshRows("history/" + encodeURIComponent(name), target, answer.pending);
  });
}

async function deleteSessions(names) {
  await run("history-error", async () => {
    const answer = await api("history/delete", { names });
    for (const key of [...openMoves]) if (names.some((name) => key.startsWith(name + "/"))) openMoves.delete(key);
    const hands = document.getElementById("session-hands");
    if (names.includes(hands.dataset.name)) {
      show("history");
      hands.replaceChildren();
      document.getElementById("history-coach-panel").hidden = true;
    }
    listSessions(answer);
  });
}

async function reviewSaved(name, hand) {
  await run("history-error", async () => {
    const panel = document.getElementById("history-coach-panel");
    const target = document.getElementById("history-coach");
    panel.hidden = false;
    tools("history-tools");
    target.replaceChildren(element("p", "waiting", "Reviewing hand " + hand + "..."));
    try {
      const answer = await api("history/" + encodeURIComponent(name) + "/review", { hand });
      target.replaceChildren(coachView(answer, true));
      copyTools("history-tools", answer);
      panel.scrollIntoView({ behavior: "smooth" });
    } catch (error) {
      target.replaceChildren();
      panel.hidden = true;
      throw error;
    }
  });
}

// Setup and page-wide controls

const setupForm = document.getElementById("setup-form");
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
fillSeats();

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
  if (options.mode === "training") {
    if (form.get("position")) options.position = form.get("position");
    if (handsSpec()) options.hands = handsSpec();
  }
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
document.getElementById("words-link").addEventListener("click", () => { document.getElementById("glossary").open = true; });
for (const link of document.querySelectorAll("a.download")) link.prepend(icon("download"));

const hideHelpers = document.getElementById("hide-helpers");
hideHelpers.checked = window.localStorage.getItem(HELPERS_KEY) === "1";
document.body.classList.toggle("no-helpers", hideHelpers.checked);
hideHelpers.addEventListener("change", () => {
  window.localStorage.setItem(HELPERS_KEY, hideHelpers.checked ? "1" : "0");
  document.body.classList.toggle("no-helpers", hideHelpers.checked);
});

const chanceBox = document.getElementById("show-chance");
chanceBox.checked = window.localStorage.getItem(CHANCE_KEY) === "1";
chanceBox.addEventListener("change", () => {
  window.localStorage.setItem(CHANCE_KEY, chanceBox.checked ? "1" : "0");
  showChance();
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
