"use strict";

// The browser side of the local web table: it renders what /api returns and sends actions.
// All text goes into the page with textContent, so nothing the server sends becomes markup.

const SUITS = { c: "♣", d: "♦", h: "♥", s: "♠" };
const PRESETS = [["1/3", 1 / 3], ["1/2", 1 / 2], ["2/3", 2 / 3], ["pot", 1]];
let session = null;
let state = null;

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const response = await fetch("/api/" + path, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function card(text) {
  const red = text[1] === "h" || text[1] === "d";
  return element("span", red ? "card red" : "card", text[0] + SUITS[text[1]]);
}

function button(label, handler, disabled) {
  const node = element("button", "", label);
  node.type = "button";
  node.disabled = Boolean(disabled);
  node.addEventListener("click", handler);
  return node;
}

function amountText(value) {
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
}

function showError(id, error) {
  document.getElementById(id).textContent = error ? error.message : "";
}

async function run(id, work) {
  showError(id, null);
  try {
    await work();
  } catch (error) {
    showError(id, error);
  }
}

function renderSeats() {
  const seats = document.getElementById("seats");
  seats.replaceChildren();
  for (const seat of state.seats) {
    const out = seat.dealt_in === false || seat.folded;
    const box = element("div", "seat" + (seat.name === "You" ? " you" : "") + (out ? " out" : ""));
    box.append(element("div", "name", seat.name + (seat.button ? "  (D)" : "")));
    let status = "Stack " + amountText(seat.stack);
    if (seat.committed) status += "  bet " + amountText(seat.committed);
    if (seat.all_in) status += "  all-in";
    if (seat.folded) status += "  folded";
    box.append(element("div", "", status));
    const cards = element("div");
    for (const text of seat.cards || []) cards.append(card(text));
    box.append(cards);
    if (seat.hud) box.append(element("div", "hud", seat.hud));
    seats.append(box);
  }
}

function you() {
  return state.seats.find((seat) => seat.name === "You");
}

function renderSizing(raise) {
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
  amount.value = slider.value = raise.min;
  slider.oninput = () => { amount.value = slider.value; };
  amount.oninput = () => { slider.value = amount.value; };
  const presets = document.getElementById("presets");
  presets.replaceChildren();
  const call = state.legal.call || 0;
  const mine = you().committed || 0;
  for (const [label, share] of PRESETS) {
    // A pot-fraction bet or raise: call first, then add that share of the pot after calling.
    const target = mine + call + share * (state.pot + call);
    // Round to the smallest chip (a 0.01 unit with 0.5/1 blinds), then trim float noise.
    const rounded = Number((Math.round(target / state.step) * state.step).toFixed(6));
    const value = Math.min(raise.max, Math.max(raise.min, rounded));
    presets.append(button(label, () => { amount.value = slider.value = value; }));
  }
}

function renderActions() {
  const actions = document.getElementById("actions");
  const between = document.getElementById("between");
  actions.replaceChildren();
  between.replaceChildren();
  if (state.your_turn) {
    const legal = state.legal;
    if (legal.fold && !legal.check) actions.append(button("Fold", () => act({ kind: "fold" })));
    if (legal.check) actions.append(button("Check", () => act({ kind: "check" })));
    if (legal.call !== null) actions.append(button("Call " + amountText(legal.call), () => act({ kind: "call" })));
    if (legal.raise) {
      const label = legal.raise.kind === "bet" ? "Bet" : "Raise to";
      actions.append(button(label, () => act({ kind: legal.raise.kind, amount: Number(document.getElementById("amount").value) })));
    }
    actions.append(button("All-in", () => act({ kind: "allin" })));
    actions.append(button("Hint", hint));
    renderSizing(legal.raise);
  } else {
    renderSizing(null);
  }
  if (state.hand_over) {
    if (state.session_over) {
      between.append(element("span", "", "The session is over. "));
    } else if (state.awaiting_rebuy) {
      between.append(element("span", "", "You are out of chips. "));
      between.append(button("Rebuy", rebuy));
    } else if (state.knocked_out) {
      between.append(element("span", "", "You are out of the tournament. "));
      between.append(button("Fast-forward to the end", next));
    } else {
      between.append(button("Next hand", next));
    }
    between.append(button("Review this hand", review));
  }
  between.append(button("New game", newGame));
}

function render() {
  renderSeats();
  const board = document.getElementById("board");
  board.replaceChildren();
  for (const text of state.board || []) board.append(card(text));
  const pot = document.getElementById("pot");
  pot.textContent = state.pot === undefined ? "" : "Pot " + amountText(state.pot) + " (" + (state.pot / state.big_blind).toFixed(1) + "bb)";
  renderActions();
  const log = document.getElementById("log");
  log.textContent = state.log.join("\n");
  log.scrollTop = log.scrollHeight;
}

async function act(action) {
  await run("table-error", async () => {
    state = await api("sessions/" + session + "/action", action);
    render();
  });
}

async function next() {
  await run("table-error", async () => {
    document.getElementById("analysis").textContent = "";
    state = await api("sessions/" + session + "/next", {});
    render();
  });
}

async function rebuy() {
  await run("table-error", async () => {
    state = await api("sessions/" + session + "/rebuy", {});
    render();
  });
}

function newGame() {
  session = null;
  state = null;
  document.getElementById("analysis").textContent = "";
  document.getElementById("table-error").textContent = "";
  document.getElementById("table").hidden = true;
  document.getElementById("setup").hidden = false;
}

async function hint() {
  await run("table-error", async () => {
    document.getElementById("analysis").textContent = "Thinking...";
    const result = await api("sessions/" + session + "/hint", {});
    document.getElementById("analysis").textContent = result.lines.join("\n");
  });
}

async function review() {
  await run("table-error", async () => {
    document.getElementById("analysis").textContent = "Reviewing (a few seconds)...";
    const result = await api("sessions/" + session + "/review", {});
    document.getElementById("analysis").textContent = result.lines.join("\n");
  });
}

document.getElementById("setup-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  const options = {
    mode: form.get("mode"),
    seats: Number(form.get("seats")),
    tier: Number(form.get("tier")),
    blinds: form.get("blinds"),
    hide_styles: form.get("hide_styles") === "on",
  };
  if (form.get("stack")) options.stack = Number(form.get("stack"));
  run("setup-error", async () => {
    const created = await api("sessions", options);
    session = created.id;
    state = created.state;
    document.getElementById("setup").hidden = true;
    document.getElementById("table").hidden = false;
    render();
  });
});
