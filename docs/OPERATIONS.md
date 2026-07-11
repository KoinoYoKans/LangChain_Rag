# 运行手册

## 上线前检查

1. 从 `.env.example` 创建部署环境的 `.env`，替换所有 `change-me` 值。
2. `POSTGRES_PASSWORD` 与 `POSTGRES_DSN` 中的密码必须一致；`REDIS_PASSWORD` 与 `REDIS_URL` 中的密码必须一致。密码包含 URL 保留字符时，必须在连接串中进行百分号编码。
3. 将 `JWT_SECRET` 设置为至少 32 字节的随机值，并设置非默认管理员密码。
4. 设置 `HOST_EMBEDDING_MODEL_PATH`、`HOST_RERANK_MODEL_PATH` 到宿主机模型目录。运行账户必须具有只读权限。
5. 将 `CORS_ALLOWED_ORIGINS` 设置为实际浏览器控制台来源的逗号分隔列表；同源部署时可以留空。
6. 保持 `API_BIND_ADDRESS=127.0.0.1`，通过受管反向代理对外提供 HTTPS。只有确有独立 API 网关时才开放 API 监听地址。
7. `TRUSTED_PROXY_CIDRS` 必须等于前端反向代理所在的边缘网 CIDR（默认 `RAG_EDGE_SUBNET`）。不要将数据网或宽泛的私网段加入该列表。

## 启动与验收

```bash
docker compose up --build -d
docker compose ps
curl --fail http://127.0.0.1:8000/health/ready
curl --fail http://127.0.0.1:8080/
```

服务状态应为：`postgres`、`redis`、`api`、`worker` 和 `frontend` 均持续运行；`api` 的健康检查通过后，前端才会启动。首次启动会创建组织和默认管理员，随后应立即修改默认管理员密码。

## 日常巡检

- 查看 `/health/ready`，它同时反映数据库、模型、向量检索组件的可用性。
- 在控制台总览查看失败入库任务；任务中心支持重试与取消。工作进程会定期重新领取中断的任务。
- 检查审计与质量页中的低可信回答、无引用回答和质量待办。
- 通过反向代理记录访问日志，按 `X-Request-ID` 关联 API、审计和上游网关日志。

## 备份与恢复

`postgres_data` 包含关系数据、向量数据、知识库权限、审计记录和会话。使用数据库一致性备份，而不是只复制数据卷：

```bash
docker compose exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB"' | gzip > rag-$(date +%F).sql.gz
```

同时备份宿主机 `storage/` 目录，它保存原始上传文件。恢复时先恢复 PostgreSQL，再恢复同一时点的 `storage/`，最后启动 API 和 worker；不要在已有数据上盲目重放备份。

## 故障处理

- API 未就绪：查看 `docker compose logs api`，通常是模型路径、数据库连接或模型提供方配置问题。
- 入库长期停留：查看 `docker compose logs worker` 与知识库队列健康；确认 Redis 可用且模型挂载可读。
- URL 导入被拒绝：仅允许公网 HTTP/HTTPS、标准端口和受限大小的 HTML/Text 响应；不应通过修改代码绕过此限制。
- 认证或跨域失败：核对反向代理的来源，以及精确的 `CORS_ALLOWED_ORIGINS` 配置。通配符不会被服务接受。
- 审计 IP 不正确：确认前端代理位于 `TRUSTED_PROXY_CIDRS` 的边缘网中。Compose 内置 Nginx 只转发直接连接地址；若前面还有企业网关，应先在 Nginx 正确配置受信 `real_ip` 链路。
- 用户修改后被登出：这是预期安全行为。密码重置、禁用、角色或部门变更会在后续请求中撤销旧 JWT，用户需要重新登录。
