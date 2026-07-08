import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ApiOutlined,
  AuditOutlined,
  BankOutlined,
  CloudUploadOutlined,
  DeleteOutlined,
  EyeOutlined,
  FileSearchOutlined,
  KeyOutlined,
  MessageOutlined,
  ReloadOutlined,
  TeamOutlined,
} from "@ant-design/icons";
import {
  App as AntApp,
  Button,
  ConfigProvider,
  Descriptions,
  Drawer,
  Flex,
  Form,
  Input,
  Layout,
  Menu,
  Modal,
  Progress,
  Select,
  Space,
  Table,
  Tabs,
  Tag,
  Typography,
  Upload,
  message,
  theme,
} from "antd";
import { QueryClient, QueryClientProvider, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import axios from "axios";
import "antd/dist/reset.css";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";
const queryClient = new QueryClient();

function App() {
  return (
    <ConfigProvider
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: {
          colorPrimary: "#2563eb",
          borderRadius: 6,
          fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        },
      }}
    >
      <AntApp>
        <QueryClientProvider client={queryClient}>
          <Shell />
        </QueryClientProvider>
      </AntApp>
    </ConfigProvider>
  );
}

function Shell() {
  const [token, setToken] = useState(localStorage.getItem("rag_token") || "");
  const [user, setUser] = useState(JSON.parse(localStorage.getItem("rag_user") || "null"));
  const [page, setPage] = useState("knowledge");
  const api = useMemo(() => buildApi(token), [token]);

  useEffect(() => {
    if (!token) return;
    api.get("/auth/me").then(({ data }) => setUser(data)).catch(() => logout());
  }, [token]);

  function onLogin(accessToken, nextUser) {
    localStorage.setItem("rag_token", accessToken);
    localStorage.setItem("rag_user", JSON.stringify(nextUser));
    setToken(accessToken);
    setUser(nextUser);
  }

  function logout() {
    localStorage.removeItem("rag_token");
    localStorage.removeItem("rag_user");
    setToken("");
    setUser(null);
  }

  if (!token) return <Login onLogin={onLogin} />;

  const menuItems = [
    { key: "knowledge", icon: <BankOutlined />, label: "知识库" },
    { key: "chat", icon: <MessageOutlined />, label: "问答" },
    { key: "users", icon: <TeamOutlined />, label: "组织用户" },
    { key: "api", icon: <ApiOutlined />, label: "开放接口" },
    { key: "audit", icon: <AuditOutlined />, label: "审计" },
  ];

  return (
    <Layout className="app-shell">
      <Layout.Sider width={232} className="app-sider">
        <div className="brand"><FileSearchOutlined /> Enterprise RAG</div>
        <Menu mode="inline" selectedKeys={[page]} items={menuItems} onClick={(item) => setPage(item.key)} />
        <div className="sider-foot">
          <div className="user-block">
            <strong>{user?.display_name || user?.email}</strong>
            <span>{user?.role} · {user?.department_id?.slice(0, 8) || "未分配部门"}</span>
          </div>
          <Button block onClick={logout}>退出登录</Button>
        </div>
      </Layout.Sider>
      <Layout.Content className="workspace">
        {page === "knowledge" && <KnowledgeWorkspace api={api} />}
        {page === "chat" && <ChatWorkspace api={api} />}
        {page === "users" && <UsersWorkspace api={api} user={user} />}
        {page === "api" && <ApiKeyWorkspace api={api} />}
        {page === "audit" && <AuditWorkspace api={api} />}
      </Layout.Content>
    </Layout>
  );
}

function Login({ onLogin }) {
  const [loading, setLoading] = useState(false);
  async function submit(values) {
    setLoading(true);
    try {
      const { data } = await axios.post(`${API_BASE}/auth/login`, values);
      onLogin(data.access_token, data.user);
    } catch (error) {
      message.error(readError(error));
    } finally {
      setLoading(false);
    }
  }
  return (
    <div className="login-screen">
      <div className="login-panel">
        <div className="login-brand"><FileSearchOutlined /> Enterprise RAG</div>
        <Form layout="vertical" initialValues={{ email: "admin@example.com", password: "admin123456" }} onFinish={submit}>
          <Form.Item name="email" label="邮箱" rules={[{ required: true }]}>
            <Input size="large" />
          </Form.Item>
          <Form.Item name="password" label="密码" rules={[{ required: true }]}>
            <Input.Password size="large" />
          </Form.Item>
          <Button type="primary" htmlType="submit" size="large" block loading={loading}>登录</Button>
        </Form>
      </div>
    </div>
  );
}

function KnowledgeWorkspace({ api }) {
  const queryClient = useQueryClient();
  const [selectedKb, setSelectedKb] = useState("");
  const [preview, setPreview] = useState(null);
  const [kbForm] = Form.useForm();
  const [urlForm] = Form.useForm();
  const kbs = useQuery({ queryKey: ["kbs"], queryFn: async () => (await api.get("/knowledge-bases")).data.items || [] });
  const activeKb = selectedKb || kbs.data?.[0]?.id || "";
  const files = useQuery({
    queryKey: ["files", activeKb],
    enabled: Boolean(activeKb),
    queryFn: async () => (await api.get(`/knowledge-bases/${activeKb}/documents`)).data.items || [],
  });
  const jobs = useQuery({
    queryKey: ["jobs", activeKb],
    enabled: Boolean(activeKb),
    refetchInterval: 2500,
    queryFn: async () => (await api.get(`/knowledge-bases/${activeKb}/ingest-jobs`)).data.items || [],
  });
  useEffect(() => {
    if (!selectedKb && kbs.data?.[0]?.id) setSelectedKb(kbs.data[0].id);
  }, [kbs.data, selectedKb]);

  const createKb = useMutation({
    mutationFn: (values) => api.post("/knowledge-bases", values),
    onSuccess: () => {
      kbForm.resetFields();
      queryClient.invalidateQueries({ queryKey: ["kbs"] });
      message.success("知识库已创建");
    },
    onError: (error) => message.error(readError(error)),
  });
  const ingestUrl = useMutation({
    mutationFn: (values) => api.post(`/knowledge-bases/${activeKb}/urls`, values),
    onSuccess: () => {
      urlForm.resetFields();
      refreshKnowledge(queryClient, activeKb);
      message.success("URL 已加入入库队列");
    },
    onError: (error) => message.error(readError(error)),
  });
  const deleteDoc = useMutation({
    mutationFn: (fileId) => api.delete(`/knowledge-bases/${activeKb}/documents/${fileId}`),
    onSuccess: () => refreshKnowledge(queryClient, activeKb),
    onError: (error) => message.error(readError(error)),
  });
  const reindexDoc = useMutation({
    mutationFn: (fileId) => api.post(`/knowledge-bases/${activeKb}/documents/${fileId}/reindex`, {}),
    onSuccess: () => {
      refreshKnowledge(queryClient, activeKb);
      message.success("已重新加入入库队列");
    },
    onError: (error) => message.error(readError(error)),
  });

  const uploadProps = {
    multiple: true,
    showUploadList: false,
    customRequest: async ({ file, onSuccess, onError }) => {
      const form = new FormData();
      form.append("file", file);
      try {
        await api.post(`/knowledge-bases/${activeKb}/documents`, form);
        refreshKnowledge(queryClient, activeKb);
        message.success(`${file.name} 已加入入库队列`);
        onSuccess();
      } catch (error) {
        message.error(readError(error));
        onError(error);
      }
    },
  };

  return (
    <section className="page-grid">
      <div className="surface narrow">
        <PageTitle title="知识库" subtitle="按部门隔离权限，统一管理文档、URL、任务和预览。" />
        <Form form={kbForm} layout="vertical" onFinish={(values) => createKb.mutate({ ...values, visibility: values.visibility || "department" })}>
          <Form.Item name="name" label="名称" rules={[{ required: true }]}>
            <Input placeholder="例如：产品文档库" />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input.TextArea rows={2} />
          </Form.Item>
          <Form.Item name="visibility" label="可见范围" initialValue="department">
            <Select options={[{ value: "department", label: "部门" }, { value: "org", label: "全组织" }, { value: "private", label: "私有" }]} />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={createKb.isPending}>创建知识库</Button>
        </Form>
        <div className="kb-list">
          {(kbs.data || []).map((item) => (
            <button className={activeKb === item.id ? "kb-item active" : "kb-item"} key={item.id} onClick={() => setSelectedKb(item.id)}>
              <strong>{item.name}</strong>
              <span>{item.visibility} · {item.id.slice(0, 8)}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="surface main">
        <Flex justify="space-between" align="center" gap={12}>
          <PageTitle title="文档管理" subtitle="支持文本、Markdown、HTML、Word、PDF 解析，PDF 可回看页文本和块定位。" />
          <Button icon={<ReloadOutlined />} onClick={() => refreshKnowledge(queryClient, activeKb)}>刷新</Button>
        </Flex>
        <Upload.Dragger {...uploadProps} disabled={!activeKb} className="upload-zone">
          <p><CloudUploadOutlined /></p>
          <p>拖拽或点击上传文档</p>
        </Upload.Dragger>
        <Form form={urlForm} className="url-form" layout="inline" onFinish={(values) => ingestUrl.mutate(values)}>
          <Form.Item name="url" rules={[{ required: true }]} className="grow">
            <Input placeholder="https://example.com/docs/page" />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={ingestUrl.isPending}>导入 URL</Button>
        </Form>
        <Table
          rowKey="id"
          size="middle"
          loading={files.isLoading}
          dataSource={files.data || []}
          columns={[
            { title: "文件", dataIndex: "filename", render: (value, row) => <Space direction="vertical" size={0}><strong>{value}</strong><span className="muted">{row.content_type}</span></Space> },
            { title: "状态", dataIndex: "status", width: 120, render: (value) => <Tag color={value === "completed" ? "green" : value === "failed" ? "red" : "blue"}>{value}</Tag> },
            { title: "分块", dataIndex: "chunk_count", width: 90 },
            { title: "更新时间", dataIndex: "updated_at", width: 190, render: formatTime },
            {
              title: "操作",
              width: 230,
              render: (_, row) => (
                <Space>
                  <Button icon={<EyeOutlined />} onClick={() => setPreview(row)}>预览</Button>
                  <Button icon={<ReloadOutlined />} onClick={() => reindexDoc.mutate(row.id)}>重建</Button>
                  <Button danger icon={<DeleteOutlined />} onClick={() => deleteDoc.mutate(row.id)} />
                </Space>
              ),
            },
          ]}
        />
        <Typography.Title level={5}>入库任务</Typography.Title>
        <Table
          rowKey="id"
          size="small"
          pagination={false}
          dataSource={jobs.data || []}
          columns={[
            { title: "来源", render: (_, row) => row.filename || row.source_uri },
            { title: "状态", dataIndex: "status", width: 120, render: (value) => <Tag>{value}</Tag> },
            { title: "进度", dataIndex: "progress", width: 180, render: (value) => <Progress percent={value} size="small" /> },
            { title: "错误", dataIndex: "error_message" },
          ]}
        />
      </div>
      <DocumentPreview api={api} kbId={activeKb} file={preview} onClose={() => setPreview(null)} />
    </section>
  );
}

function DocumentPreview({ api, kbId, file, onClose }) {
  const preview = useQuery({
    queryKey: ["preview", kbId, file?.id],
    enabled: Boolean(kbId && file?.id),
    queryFn: async () => (await api.get(`/knowledge-bases/${kbId}/documents/${file.id}/preview`)).data,
  });
  return (
    <Drawer width="72vw" title={file?.filename} open={Boolean(file)} onClose={onClose}>
      {preview.data && (
        <Tabs
          items={[
            {
              key: "pages",
              label: "页面预览",
              children: (
                <div className="page-preview">
                  {preview.data.pages.map((page) => (
                    <article className="page-text" key={page.id}>
                      <Flex justify="space-between">
                        <strong>第 {page.page_number} 页</strong>
                        <Tag>{page.ocr_status}</Tag>
                      </Flex>
                      <pre>{page.text}</pre>
                    </article>
                  ))}
                  {!preview.data.pages.length && <EmptyText text="旧文档暂无预览数据，可点击重建生成页面定位。" />}
                </div>
              ),
            },
            {
              key: "chunks",
              label: "分块管理",
              children: (
                <Table
                  rowKey="id"
                  dataSource={preview.data.chunks}
                  columns={[
                    { title: "#", dataIndex: "chunk_index", width: 72 },
                    { title: "页码", render: (_, row) => row.location?.page_number || "-" },
                    { title: "关键词", dataIndex: "keywords", render: (values = []) => values.slice(0, 8).map((item) => <Tag key={item}>{item}</Tag>) },
                    { title: "内容", dataIndex: "content", render: (value) => <Typography.Paragraph ellipsis={{ rows: 4, expandable: true }}>{value}</Typography.Paragraph> },
                  ]}
                />
              ),
            },
            {
              key: "meta",
              label: "元数据",
              children: <Descriptions bordered column={1} items={Object.entries(preview.data.file).map(([key, value]) => ({ key, label: key, children: String(value ?? "") }))} />,
            },
          ]}
        />
      )}
    </Drawer>
  );
}

function ChatWorkspace({ api }) {
  const [kbId, setKbId] = useState("");
  const [conversationId, setConversationId] = useState("");
  const [messages, setMessages] = useState([]);
  const [form] = Form.useForm();
  const kbs = useQuery({ queryKey: ["kbs"], queryFn: async () => (await api.get("/knowledge-bases")).data.items || [] });
  useEffect(() => {
    if (!kbId && kbs.data?.[0]?.id) setKbId(kbs.data[0].id);
  }, [kbs.data, kbId]);
  const ask = useMutation({
    mutationFn: (values) => api.post("/chat", { knowledge_base_id: kbId, conversation_id: conversationId || undefined, message: values.message }),
    onSuccess: ({ data }, values) => {
      setConversationId(data.conversation_id);
      setMessages((items) => [{ question: values.message, ...data }, ...items]);
      form.resetFields();
    },
    onError: (error) => message.error(readError(error)),
  });
  return (
    <section className="surface full">
      <PageTitle title="知识库问答" subtitle="多轮改写、混合检索、重排和引用来源会一起参与回答。" />
      <Form form={form} layout="inline" className="chat-form" onFinish={(values) => ask.mutate(values)}>
        <Form.Item className="kb-select">
          <Select value={kbId} onChange={setKbId} options={(kbs.data || []).map((item) => ({ value: item.id, label: item.name }))} />
        </Form.Item>
        <Form.Item name="message" rules={[{ required: true }]} className="grow">
          <Input.Search placeholder="输入问题" enterButton="发送" loading={ask.isPending} />
        </Form.Item>
      </Form>
      <div className="answer-list">
        {messages.map((item) => (
          <article className="answer-item" key={item.assistant_message_id}>
            <Typography.Title level={5}>{item.question}</Typography.Title>
            <Typography.Paragraph>{item.answer}</Typography.Paragraph>
            <Table
              rowKey={(row, index) => `${row.chunk_id}-${index}`}
              size="small"
              pagination={false}
              dataSource={item.sources}
              columns={[
                { title: "文件", dataIndex: "filename" },
                { title: "分块", dataIndex: "chunk_index", width: 80 },
                { title: "页码", dataIndex: "page_number", width: 80, render: (value) => value || "-" },
                { title: "分数", width: 140, render: (_, row) => scoreText(row.rerank_score ?? row.hybrid_score ?? row.vector_score) },
                { title: "片段", dataIndex: "snippet" },
              ]}
            />
          </article>
        ))}
      </div>
    </section>
  );
}

function UsersWorkspace({ api, user }) {
  const queryClient = useQueryClient();
  const [deptForm] = Form.useForm();
  const [userForm] = Form.useForm();
  const departments = useQuery({ queryKey: ["departments"], queryFn: async () => (await api.get("/departments")).data.items || [] });
  const users = useQuery({ queryKey: ["users"], enabled: user?.role !== "member", queryFn: async () => (await api.get("/users")).data.items || [] });
  const createDept = useMutation({
    mutationFn: (values) => api.post("/departments", values),
    onSuccess: () => {
      deptForm.resetFields();
      queryClient.invalidateQueries({ queryKey: ["departments"] });
    },
    onError: (error) => message.error(readError(error)),
  });
  const createUser = useMutation({
    mutationFn: (values) => api.post("/users", { ...values, department_id: values.department_id || null }),
    onSuccess: () => {
      userForm.resetFields();
      queryClient.invalidateQueries({ queryKey: ["users"] });
    },
    onError: (error) => message.error(readError(error)),
  });
  return (
    <section className="page-grid">
      <div className="surface narrow">
        <PageTitle title="部门" subtitle="保留父部门字段，知识库权限可以按部门归属控制。" />
        <Form form={deptForm} layout="vertical" onFinish={(values) => createDept.mutate(values)}>
          <Form.Item name="name" label="部门名称" rules={[{ required: true }]}><Input /></Form.Item>
          <Form.Item name="parent_id" label="上级部门"><Select allowClear options={(departments.data || []).map((item) => ({ value: item.id, label: item.name }))} /></Form.Item>
          <Button type="primary" htmlType="submit" loading={createDept.isPending}>创建部门</Button>
        </Form>
        <Table rowKey="id" size="small" pagination={false} dataSource={departments.data || []} columns={[{ title: "名称", dataIndex: "name" }, { title: "上级", dataIndex: "parent_id", render: (value) => value?.slice(0, 8) || "-" }]} />
      </div>
      <div className="surface main">
        <PageTitle title="用户" subtitle="记录具体用户、角色、部门和最近登录，用于审计和权限判断。" />
        <Form form={userForm} layout="inline" className="user-create" onFinish={(values) => createUser.mutate(values)}>
          <Form.Item name="email" rules={[{ required: true }]}><Input placeholder="邮箱" /></Form.Item>
          <Form.Item name="display_name" rules={[{ required: true }]}><Input placeholder="姓名" /></Form.Item>
          <Form.Item name="password" rules={[{ required: true, min: 8 }]}><Input.Password placeholder="密码" /></Form.Item>
          <Form.Item name="role" initialValue="member"><Select options={["member", "manager", "admin"].map((value) => ({ value, label: value }))} /></Form.Item>
          <Form.Item name="department_id"><Select allowClear placeholder="部门" options={(departments.data || []).map((item) => ({ value: item.id, label: item.name }))} /></Form.Item>
          <Button type="primary" htmlType="submit" loading={createUser.isPending}>创建用户</Button>
        </Form>
        <Table
          rowKey="id"
          dataSource={users.data || []}
          columns={[
            { title: "用户", render: (_, row) => <Space direction="vertical" size={0}><strong>{row.display_name}</strong><span className="muted">{row.email}</span></Space> },
            { title: "角色", dataIndex: "role", render: (value) => <Tag color={value === "admin" ? "red" : value === "manager" ? "gold" : "blue"}>{value}</Tag> },
            { title: "部门", dataIndex: "department_id", render: (value) => departments.data?.find((item) => item.id === value)?.name || value?.slice(0, 8) || "-" },
            { title: "最近登录", dataIndex: "last_login_at", render: formatTime },
            { title: "创建时间", dataIndex: "created_at", render: formatTime },
          ]}
        />
      </div>
    </section>
  );
}

function ApiKeyWorkspace({ api }) {
  const queryClient = useQueryClient();
  const [form] = Form.useForm();
  const [secret, setSecret] = useState("");
  const kbs = useQuery({ queryKey: ["kbs"], queryFn: async () => (await api.get("/knowledge-bases")).data.items || [] });
  const keys = useQuery({ queryKey: ["apiKeys"], queryFn: async () => (await api.get("/api-keys")).data.items || [] });
  const createKey = useMutation({
    mutationFn: (values) => api.post("/api-keys", values),
    onSuccess: ({ data }) => {
      setSecret(data.secret);
      form.resetFields();
      queryClient.invalidateQueries({ queryKey: ["apiKeys"] });
    },
    onError: (error) => message.error(readError(error)),
  });
  return (
    <section className="surface full">
      <PageTitle title="开放接口" subtitle="为其他项目预留 OpenAI 兼容问答接口和独立检索接口，Key 绑定具体用户与知识库。" />
      <Form form={form} layout="inline" onFinish={(values) => createKey.mutate(values)}>
        <Form.Item name="name" rules={[{ required: true }]}><Input prefix={<KeyOutlined />} placeholder="Key 名称" /></Form.Item>
        <Form.Item name="knowledge_base_id" rules={[{ required: true }]} className="kb-select">
          <Select placeholder="绑定知识库" options={(kbs.data || []).map((item) => ({ value: item.id, label: item.name }))} />
        </Form.Item>
        <Button type="primary" htmlType="submit" loading={createKey.isPending}>创建 API Key</Button>
      </Form>
      <Descriptions className="api-doc" bordered column={1} items={[
        { key: "chat", label: "OpenAI 兼容", children: "POST /api/v1/chat/completions，Header: Authorization: Bearer rag-..." },
        { key: "retrieval", label: "检索接口", children: "POST /api/v1/knowledge/{knowledge_base_id}/retrieval" },
      ]} />
      <Table
        rowKey="id"
        dataSource={keys.data || []}
        columns={[
          { title: "名称", dataIndex: "name" },
          { title: "前缀", dataIndex: "key_prefix" },
          { title: "知识库", dataIndex: "knowledge_base_id", render: (value) => kbs.data?.find((item) => item.id === value)?.name || value?.slice(0, 8) },
          { title: "最近使用", dataIndex: "last_used_at", render: formatTime },
          { title: "创建时间", dataIndex: "created_at", render: formatTime },
        ]}
      />
      <Modal title="API Key 只显示一次" open={Boolean(secret)} footer={<Button type="primary" onClick={() => setSecret("")}>我已保存</Button>} onCancel={() => setSecret("")}>
        <Input.TextArea value={secret} autoSize readOnly />
      </Modal>
    </section>
  );
}

function AuditWorkspace({ api }) {
  const logs = useQuery({ queryKey: ["audit"], refetchInterval: 5000, queryFn: async () => (await api.get("/audit-logs")).data.items || [] });
  return (
    <section className="surface full">
      <PageTitle title="审计日志" subtitle="记录用户、部门、IP、请求 ID、操作对象、结果和耗时。" />
      <Table
        rowKey="id"
        loading={logs.isLoading}
        dataSource={logs.data || []}
        expandable={{ expandedRowRender: (row) => <pre className="json-block">{JSON.stringify(row, null, 2)}</pre> }}
        columns={[
          { title: "操作", dataIndex: "action" },
          { title: "结果", dataIndex: "result", width: 100, render: (value) => <Tag color={value === "success" ? "green" : "red"}>{value}</Tag> },
          { title: "用户", dataIndex: "actor_user_id", render: (value) => value?.slice(0, 8) || "-" },
          { title: "部门", dataIndex: "actor_department_id", render: (value) => value?.slice(0, 8) || "-" },
          { title: "IP", dataIndex: "ip_address" },
          { title: "对象", render: (_, row) => `${row.target_type || "-"} / ${row.target_id || "-"}` },
          { title: "耗时", dataIndex: "latency_ms", render: (value) => value == null ? "-" : `${value}ms` },
          { title: "时间", dataIndex: "created_at", render: formatTime },
        ]}
      />
    </section>
  );
}

function PageTitle({ title, subtitle }) {
  return (
    <div className="page-title">
      <Typography.Title level={3}>{title}</Typography.Title>
      <Typography.Text type="secondary">{subtitle}</Typography.Text>
    </div>
  );
}

function EmptyText({ text }) {
  return <div className="empty-text">{text}</div>;
}

function buildApi(token) {
  const instance = axios.create({ baseURL: API_BASE });
  instance.interceptors.request.use((config) => {
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  });
  return instance;
}

function refreshKnowledge(queryClient, kbId) {
  queryClient.invalidateQueries({ queryKey: ["files", kbId] });
  queryClient.invalidateQueries({ queryKey: ["jobs", kbId] });
}

function readError(error) {
  const detail = error?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (detail?.message) return detail.message;
  return error?.message || "请求失败";
}

function formatTime(value) {
  if (!value) return "-";
  return new Date(value).toLocaleString();
}

function scoreText(value) {
  if (value == null) return "-";
  return Number(value).toFixed(3);
}

createRoot(document.getElementById("root")).render(<App />);
