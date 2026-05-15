#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { parseArgs } from 'node:util'

const USAGE =
  'Usage: npm run dev:remote -- <user>@<host>:<ssh_port> [<backend_port>] [<local_port>]\n' +
  '  example: npm run dev:remote -- root@194.68.245.214:22036\n' +
  '  example: npm run dev:remote -- root@194.68.245.214:22036 9000'

const { positionals } = parseArgs({ allowPositionals: true, options: {} })

const target = positionals[0]
if (!target) {
  console.error(USAGE)
  process.exit(2)
}

const m = target.match(/^([^@]+)@([^:]+):(\d+)$/)
if (!m) {
  console.error(`Invalid target "${target}". Expected user@host:ssh_port`)
  console.error(USAGE)
  process.exit(2)
}
const [, user, host, sshPort] = m
const backendPort = positionals[1] ?? '8000'
const localPort = positionals[2] ?? '8000'

const sshArgs = [
  '-N',
  '-o', 'ExitOnForwardFailure=yes',
  '-o', 'ServerAliveInterval=30',
  '-o', 'ServerAliveCountMax=3',
  '-L', `${localPort}:127.0.0.1:${backendPort}`,
  '-p', sshPort,
  `${user}@${host}`,
]
console.log(`[dev-remote] ssh ${sshArgs.join(' ')}`)

const isWin = process.platform === 'win32'
const ssh = spawn('ssh', sshArgs, { stdio: 'inherit' })
const vite = spawn('npm', ['run', 'dev'], { stdio: 'inherit', shell: true })

function killTree(child) {
  if (!child || child.exitCode !== null) return
  if (isWin) {
    spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' })
  } else {
    child.kill('SIGTERM')
  }
}

let shuttingDown = false
function shutdown(code) {
  if (shuttingDown) return
  shuttingDown = true
  killTree(ssh)
  killTree(vite)
  setTimeout(() => process.exit(code ?? 0), 300)
}

ssh.on('exit', (code) => {
  if (!shuttingDown) console.error(`[dev-remote] ssh exited (${code}). shutting down vite.`)
  shutdown(code ?? 1)
})
vite.on('exit', (code) => {
  if (!shuttingDown) console.log(`[dev-remote] vite exited (${code}). shutting down ssh.`)
  shutdown(code ?? 0)
})

process.on('SIGINT', () => shutdown(0))
process.on('SIGTERM', () => shutdown(0))
