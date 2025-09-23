<template>
  <div class="min-h-screen bg-slate-50 text-slate-900">
    <!-- Header -->
    <header class="bg-gradient-to-r from-indigo-600 via-violet-600 to-fuchsia-600 text-white">
      <div class="mx-auto max-w-5xl px-4 py-8">
        <h1 class="text-2xl font-semibold tracking-tight">Quotes RAG — Tailwind UI</h1>
        <p class="mt-1 text-sm/6 text-white/80">
          Ask a question, get an answer with citations. Powered by your pgvector+FastAPI service.
        </p>
      </div>
    </header>

    <!-- Main -->
    <main class="mx-auto max-w-5xl px-4 py-8">
      <!-- Card: Query -->
      <section class="rounded-2xl bg-white shadow-sm ring-1 ring-black/5">
        <div class="grid gap-4 border-b border-slate-100 p-4 sm:grid-cols-[1fr_auto_auto_auto] sm:items-end">
          <div class="sm:col-span-1">
            <label class="mb-1 block text-sm font-medium text-slate-700">Your question</label>
            <input
              v-model="q"
              type="text"
              placeholder="e.g., what is happiness"
              class="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-slate-900 outline-none ring-indigo-500/0 transition focus:ring-2"
            />
          </div>

          <div class="grid gap-1 sm:ml-4">
            <label class="text-sm font-medium text-slate-700">Top-k</label>
            <input
              v-model.number="k"
              type="number"
              min="1" max="50"
              class="w-28 rounded-lg border border-slate-200 bg-white px-3 py-2 text-slate-900 outline-none ring-indigo-500/0 transition focus:ring-2"
            />
          </div>

          <div class="grid gap-1 sm:ml-2">
            <label class="text-sm font-medium text-slate-700">Max context chars</label>
            <input
              v-model.number="maxCtx"
              type="number"
              min="200" max="4000" step="50"
              class="w-36 rounded-lg border border-slate-200 bg-white px-3 py-2 text-slate-900 outline-none ring-indigo-500/0 transition focus:ring-2"
            />
          </div>

          <div class="grid gap-1 sm:ml-2">
            <label class="text-sm font-medium text-slate-700">Max new tokens</label>
            <input
              v-model.number="maxNew"
              type="number"
              min="32" max="512" step="8"
              class="w-36 rounded-lg border border-slate-200 bg-white px-3 py-2 text-slate-900 outline-none ring-indigo-500/0 transition focus:ring-2"
            />
          </div>

          <div class="sm:col-span-1"></div>
        </div>

        <div class="flex flex-wrap items-center gap-2 p-4">
          <button
            :disabled="!canAsk"
            @click="ask"
            class="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <svg v-if="loading" class="size-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
              <path class="opacity-75" fill="currentColor"
                    d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/>
            </svg>
            <span>{{ loading ? 'Asking…' : 'Ask' }}</span>
          </button>

          <button
            :disabled="!canAsk || loading"
            @click="searchOnly"
            class="inline-flex items-center gap-2 rounded-lg bg-white px-4 py-2 text-slate-900 ring-1 ring-inset ring-slate-200 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
          >
            Preview retrieval
          </button>

          <span v-if="error" class="ml-1 rounded-md bg-rose-50 px-2 py-1 text-sm font-medium text-rose-700 ring-1 ring-rose-200">
            {{ error }}
          </span>
        </div>
      </section>

      <!-- Card: Answer -->
      <section v-if="answer" class="mt-6 rounded-2xl bg-white p-4 shadow-sm ring-1 ring-black/5">
        <div class="mb-2 flex items-center justify-between">
          <h2 class="text-base font-semibold text-slate-800">Answer</h2>
          <span class="rounded-full bg-indigo-50 px-2 py-1 text-xs font-medium text-indigo-700 ring-1 ring-indigo-200">
            Generated
          </span>
        </div>
        <div class="prose max-w-none prose-p:my-0 prose-pre:my-0 text-slate-800">
          <pre class="whitespace-pre-wrap break-words text-[15px] leading-6">{{ answer }}</pre>
        </div>
      </section>

      <!-- Card: Citations -->
      <section v-if="citations.length" class="mt-6 rounded-2xl bg-white p-4 shadow-sm ring-1 ring-black/5">
        <h2 class="mb-3 text-base font-semibold text-slate-800">Citations</h2>
        <ol class="space-y-3 pl-4">
          <li
            v-for="(c, idx) in citations"
            :key="c.id"
            class="rounded-xl border border-slate-200/60 bg-slate-50 p-3"
          >
            <div class="mb-1 flex items-center gap-2">
              <span class="inline-flex size-6 items-center justify-center rounded-full bg-indigo-600 text-xs font-bold text-white">
                {{ idx + 1 }}
              </span>
              <span class="text-sm font-medium text-slate-700">{{ c.author }}</span>
              <span class="text-xs text-slate-500">· score {{ c.score.toFixed(3) }}</span>
            </div>
            <p class="text-[15px] leading-6 text-slate-800">{{ c.text }}</p>
            <!-- score bar -->
            <div class="mt-2 h-1.5 w-full rounded-full bg-slate-200">
              <div
                class="h-1.5 rounded-full bg-indigo-600 transition-all"
                :style="{ width: Math.min(100, Math.max(0, Math.round(c.score * 100))) + '%' }"
              ></div>
            </div>
          </li>
        </ol>
      </section>

      <!-- Card: Retrieval preview -->
      <section v-if="results.length" class="mt-6 rounded-2xl bg-white p-4 shadow-sm ring-1 ring-black/5">
        <h2 class="mb-3 text-base font-semibold text-slate-800">Retrieved (search preview)</h2>
        <ol class="space-y-3 pl-4">
          <li
            v-for="(r, idx) in results"
            :key="r.id"
            class="rounded-xl border border-slate-200/60 bg-white p-3"
          >
            <div class="mb-1 flex items-center gap-2">
              <span class="inline-flex size-6 items-center justify-center rounded-full bg-slate-800 text-xs font-bold text-white">
                {{ idx + 1 }}
              </span>
              <span class="text-sm font-medium text-slate-700">{{ r.author }}</span>
              <span class="text-xs text-slate-500">· score {{ r.score.toFixed(3) }}</span>
            </div>
            <p class="text-[15px] leading-6 text-slate-800">{{ r.text }}</p>
          </li>
        </ol>
      </section>
    </main>
  </div>
</template>

<script setup lang="js">
import { ref, computed } from 'vue'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

const q = ref('')
const k = ref(5)
const maxCtx = ref(1200)
const maxNew = ref(180)

const loading = ref(false)
const error = ref('')
const answer = ref('')
const citations = ref([])
const results = ref([])

const canAsk = computed(() => q.value.trim().length > 0 && !loading.value)

async function ask () {
  error.value = ''
  answer.value = ''
  citations.value = []
  results.value = []
  loading.value = true
  try {
    const params = new URLSearchParams({
      q: q.value,
      k: String(k.value),
      max_ctx_chars: String(maxCtx.value),
      max_new_tokens: String(maxNew.value),
    })
    const res = await fetch(`${API_BASE}/ask?${params}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    answer.value = data.answer || ''
    citations.value = Array.isArray(data.citations) ? data.citations : []
  } catch (e) {
    error.value = e?.message || String(e)
  } finally {
    loading.value = false
  }
}

async function searchOnly () {
  error.value = ''
  answer.value = ''
  citations.value = []
  results.value = []
  loading.value = true
  try {
    const params = new URLSearchParams({ q: q.value, k: String(k.value) })
    const res = await fetch(`${API_BASE}/search?${params}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    results.value = Array.isArray(data.results) ? data.results : []
  } catch (e) {
    error.value = e?.message || String(e)
  } finally {
    loading.value = false
  }
}
</script>
