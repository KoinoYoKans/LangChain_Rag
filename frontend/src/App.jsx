import React, { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ApiOutlined,
  ArrowRightOutlined,
  AuditOutlined,
  BankOutlined,
  CloudUploadOutlined,
  DashboardOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  DislikeOutlined,
  EditOutlined,
  EyeOutlined,
  FileSearchOutlined,
  HistoryOutlined,
  LikeOutlined,
  MessageOutlined,
  ReloadOutlined,
  SafetyCertificateOutlined,
  TeamOutlined,
  WarningOutlined,
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
  InputNumber,
  Layout,
  Menu,
  Modal,
  Progress,
  Select,
  Space,
  Switch,
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
const ApiKeyWorkspace = lazy(() => import("./workspaces/ApiKeyWorkspace"));

function App() {
  return (
    <ConfigProvider
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: {
          colorPrimary: "#0f766e",
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
  const [page, setPage] = useState("overview");
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
    { key: "overview", icon: <DashboardOutlined />, label: "运营总览" },
    { key: "knowledge", icon: <BankOutlined />, label: "知识库" },
    { key: "chat", icon: <MessageOutlined />, label: "问答" },
    ...(user?.role === "admin" || user?.role === "manager"
      ? [
          { key: "users", icon: <TeamOutlined />, label: "组织用户" },
          { key: "api", icon: <ApiOutlined />, label: "开放接口" },
          { key: "audit", icon: <AuditOutlined />, label: "审计与质量" },
        ]
      : []),
  ];

  return (
    <Layout className="app-shell">
      <Layout.Sider width={232} className="app-sider">
        <div className="brand"><FileSearchOutlined /> Enterprise RAG</div>
        <Menu mode="inline" selectedKeys={[page]} items={menuItems} onClick={(item) => setPage(item.key)} />
        <div className="sider-foot">
          <div className="user-block">
            <strong>{user?.display_name || user?.email}</strong>
            <span>{user?.role === "admin" ? "管理员" : user?.role === "manager" ? "经理" : "成员"} · {user?.department_id?.slice(0, 8) || "未分配部门"}</span>
          </div>
          <Button block onClick={logout}>退出登录</Button>
        </div>
      </Layout.Sider>
      <Layout.Content className="workspace">
        {page === "overview" && <OverviewWorkspace api={api} user={user} onNavigate={setPage} />}
        {page === "knowledge" && <KnowledgeWorkspace api={api} user={user} />}
        {page === "chat" && <ChatWorkspace api={api} />}
        {page === "users" && <UsersWorkspace api={api} user={user} />}
        {page === "api" && <Suspense fallback={<WorkspaceLoading />}><ApiKeyWorkspace api={api} /></Suspense>}
        {page === "audit" && <AuditWorkspace api={api} user={user} />}
      </Layout.Content>
    </Layout>
  );
}

function OverviewWorkspace({ api, user, onNavigate }) {
  const knowledgeBases = useQuery({
    queryKey: ["overview-kbs"],
    queryFn: async () => (await api.get("/knowledge-bases")).data.items || [],
  });
  const serviceHealth = useQuery({
    queryKey: ["service-health"],
    queryFn: async () => (await api.get("/health")).data,
    refetchInterval: 30_000,
    retry: false,
  });
  const items = knowledgeBases.data || [];
  const accessible = items.filter((item) => item.can_read);
  const writable = accessible.filter((item) => item.can_write);
  const documents = accessible.reduce((total, item) => total + Number(item.file_count || 0), 0);
  const completed = accessible.reduce((total, item) => total + Number(item.completed_file_count || 0), 0);
  const failedJobs = accessible.reduce((total, item) => total + Number(item.failed_job_count || 0), 0);
  const attention = accessible.filter((item) => Number(item.failed_job_count || 0) > 0 || (Number(item.file_count || 0) > 0 && Number(item.completed_file_count || 0) === 0));
  const healthState = serviceHealth.isError ? "unknown" : serviceHealth.data?.ready ? "ready" : "degraded";
  const healthLabel = healthState === "ready" ? "服务就绪" : healthState === "degraded" ? "依赖检查中" : "状态未知";

  return (
    <section className="overview-page">
      <header className="overview-header">
        <div>
          <span className="eyebrow">{user?.role === "admin" ? "管理视图" : user?.role === "manager" ? "运营视图" : "成员视图"}</span>
          <Typography.Title level={2}>运营总览</Typography.Title>
          <Typography.Text type="secondary">知识资产、处理状态与待关注事项。</Typography.Text>
        </div>
        <Space wrap>
          <span className={`service-status ${healthState}`} title="API、数据库与检索模型的当前就绪状态">
            <SafetyCertificateOutlined /> {healthLabel}
          </span>
          <Button icon={<MessageOutlined />} onClick={() => onNavigate("chat")}>进入问答</Button>
          {writable.length > 0 && <Button type="primary" icon={<CloudUploadOutlined />} onClick={() => onNavigate("knowledge")}>管理知识库</Button>}
        </Space>
      </header>

      <div className="metric-grid" aria-label="知识库运营指标">
        <MetricCard icon={<DatabaseOutlined />} label="可访问知识库" value={accessible.length} tone="teal" />
        <MetricCard icon={<FileSearchOutlined />} label="已登记文档" value={documents} tone="blue" />
        <MetricCard icon={<SafetyCertificateOutlined />} label="完成入库" value={completed} tone="green" />
        <MetricCard icon={<WarningOutlined />} label="待处理失败任务" value={failedJobs} tone={failedJobs ? "red" : "amber"} />
      </div>

      <div className="overview-grid">
        <section className="surface overview-section">
          <div className="section-heading">
            <div>
              <Typography.Title level={4}>知识库状态</Typography.Title>
              <Typography.Text type="secondary">按当前权限范围汇总。</Typography.Text>
            </div>
            <Button type="link" icon={<ArrowRightOutlined />} iconPosition="end" onClick={() => onNavigate("knowledge")}>查看知识库</Button>
          </div>
          <Table
            rowKey="id"
            loading={knowledgeBases.isLoading}
            dataSource={accessible.slice(0, 8)}
            pagination={false}
            scroll={{ x: "max-content" }}
            size="middle"
            locale={{ emptyText: "暂无可访问知识库" }}
            columns={[
              { title: "知识库", dataIndex: "name", render: (value, row) => <div className="table-primary"><strong>{value}</strong><span>{row.description || "未填写说明"}</span></div> },
              { title: "可用文档", width: 110, render: (_, row) => `${row.completed_file_count || 0}/${row.file_count || 0}` },
              { title: "失败任务", dataIndex: "failed_job_count", width: 110, render: (value) => value ? <Tag color="error">{value}</Tag> : <Tag color="success">0</Tag> },
              { title: "权限", width: 110, render: (_, row) => <Tag color={row.can_write ? "processing" : "default"}>{row.can_write ? "可维护" : "只读"}</Tag> },
            ]}
          />
        </section>
        <section className="surface overview-section attention-panel">
          <div className="section-heading">
            <div>
              <Typography.Title level={4}>待关注</Typography.Title>
              <Typography.Text type="secondary">优先处理入库失败和未完成的数据源。</Typography.Text>
            </div>
            <WarningOutlined className="section-heading-icon" />
          </div>
          {attention.length ? (
            <div className="attention-list">
              {attention.slice(0, 6).map((item) => (
                <button className="attention-item" key={item.id} onClick={() => onNavigate("knowledge")}>
                  <span className="attention-status"><WarningOutlined /></span>
                  <span><strong>{item.name}</strong><small>{item.failed_job_count ? `${item.failed_job_count} 个失败任务` : "文档尚未完成入库"}</small></span>
                  <ArrowRightOutlined />
                </button>
              ))}
            </div>
          ) : <EmptyText text="当前没有需要处理的知识库任务" />}
        </section>
      </div>
    </section>
  );
}

function MetricCard({ icon, label, value, tone }) {
  return (
    <section className={`metric-card metric-${tone}`}>
      <span className="metric-icon">{icon}</span>
      <div><span>{label}</span><strong>{value}</strong></div>
    </section>
  );
}

function Login({ onLogin }) {
  const [loading, setLoading] = useState(false);
  const [errorText, setErrorText] = useState("");
  async function submit(values) {
    setLoading(true);
    setErrorText("");
    try {
      const { data } = await axios.post(`${API_BASE}/auth/login`, values);
      onLogin(data.access_token, data.user);
    } catch (error) {
      const detail = readError(error);
      setErrorText(detail);
      message.error(detail);
    } finally {
      setLoading(false);
    }
  }
  return (
    <div className="login-screen">
      <div className="login-panel">
        <div className="login-brand"><FileSearchOutlined /> Enterprise RAG</div>
        <Form layout="vertical" onFinish={submit} onValuesChange={() => setErrorText("")}>
          <Form.Item name="email" label="邮箱" rules={[{ required: true }, { type: "email", message: "请输入有效的邮箱地址" }]}>
            <Input size="large" placeholder="admin@example.com" />
          </Form.Item>
          <Form.Item name="password" label="密码" rules={[{ required: true }]}>
            <Input.Password size="large" placeholder="输入密码" />
          </Form.Item>
          {errorText && <div className="login-error" role="alert">{errorText}</div>}
          <Button type="primary" htmlType="submit" size="large" block loading={loading}>登录</Button>
        </Form>
      </div>
    </div>
  );
}

function KnowledgeWorkspace({ api, user }) {
  const queryClient = useQueryClient();
  const [selectedKb, setSelectedKb] = useState("");
  const [preview, setPreview] = useState(null);
  const [editingKb, setEditingKb] = useState(null);
  const [fileStatus, setFileStatus] = useState("");
  const [taskStatus, setTaskStatus] = useState("all");
  const [taskDetail, setTaskDetail] = useState(null);
  const [selectedFileIds, setSelectedFileIds] = useState([]);
  const [selectedJobIds, setSelectedJobIds] = useState([]);
  const [urlPlan, setUrlPlan] = useState(null);
  const [selectedUrlItemIds, setSelectedUrlItemIds] = useState([]);
  const [kbForm] = Form.useForm();
  const [editForm] = Form.useForm();
  const [memberForm] = Form.useForm();
  const [urlForm] = Form.useForm();
  const grantSubjectType = Form.useWatch("subject_type", memberForm) || "user";
  const kbs = useQuery({ queryKey: ["kbs"], queryFn: async () => (await api.get("/knowledge-bases")).data.items || [] });
  const fullAccessKbs = (kbs.data || []).filter(isFullAccessKb);
  const departments = useQuery({ queryKey: ["departments"], queryFn: async () => (await api.get("/departments")).data.items || [] });
  const activeKb = selectedKb || fullAccessKbs[0]?.id || "";
  const activeKbRecord = fullAccessKbs.find((item) => item.id === activeKb);
  const canWriteKb = Boolean(activeKbRecord?.can_write);
  const canManageKb = Boolean(activeKbRecord?.has_full_access);
  const canManageMembers = Boolean(activeKbRecord?.has_full_access);
  const memberCandidates = useQuery({
    queryKey: ["kb-member-candidates", activeKb],
    enabled: Boolean(activeKb && activeKbRecord?.can_manage_members),
    queryFn: async () => (await api.get(`/knowledge-bases/${activeKb}/member-candidates`)).data.items || [],
  });
  const grants = useQuery({
    queryKey: ["kb-grants", activeKb],
    enabled: Boolean(activeKb),
    queryFn: async () => (await api.get(`/knowledge-bases/${activeKb}/grants`)).data.items || [],
  });
  const files = useQuery({
    queryKey: ["files", activeKb, fileStatus],
    enabled: Boolean(activeKb),
    queryFn: async () => (await api.get(`/knowledge-bases/${activeKb}/documents`, { params: fileStatus ? { status: fileStatus } : {} })).data.items || [],
  });
  const documentTasks = useQuery({
    queryKey: ["document-tasks", activeKb, taskStatus],
    enabled: Boolean(activeKb),
    refetchInterval: (query) => hasActiveIngestWork(query.state.data) ? 2500 : false,
    queryFn: async () => (await api.get(`/knowledge-bases/${activeKb}/document-tasks`, { params: { status: taskStatus } })).data.items || [],
  });
  const activeJobs = useQuery({
    queryKey: ["jobs", activeKb, "active"],
    enabled: Boolean(activeKb),
    refetchInterval: (query) => hasActiveIngestWork(query.state.data) ? 2500 : false,
    queryFn: async () => (await api.get(`/knowledge-bases/${activeKb}/ingest-jobs`, { params: { status: "active" } })).data.items || [],
  });
  const historyJobs = useQuery({
    queryKey: ["jobs", activeKb, "history"],
    enabled: Boolean(activeKb),
    queryFn: async () => (await api.get(`/knowledge-bases/${activeKb}/ingest-jobs`, { params: { status: "history" } })).data.items || [],
  });
  const queueHealth = useQuery({
    queryKey: ["queue-health", activeKb],
    enabled: Boolean(activeKb),
    refetchInterval: (query) => (query.state.data?.pending_count || query.state.data?.running_count) ? 2500 : false,
    queryFn: async () => (await api.get(`/knowledge-bases/${activeKb}/queue-health`)).data,
  });
  useEffect(() => {
    if (!kbs.data) return;
    if (!selectedKb && fullAccessKbs[0]?.id) {
      setSelectedKb(fullAccessKbs[0].id);
      return;
    }
    if (selectedKb && !fullAccessKbs.some((item) => item.id === selectedKb)) {
      setSelectedKb(fullAccessKbs[0]?.id || "");
    }
  }, [kbs.data, selectedKb, fullAccessKbs]);
  useEffect(() => {
    setUrlPlan(null);
    setSelectedUrlItemIds([]);
    urlForm.resetFields(["urls"]);
  }, [activeKb, urlForm]);

  const createKb = useMutation({
    mutationFn: (values) => api.post("/knowledge-bases", values),
    onSuccess: () => {
      kbForm.resetFields();
      queryClient.invalidateQueries({ queryKey: ["kbs"] });
      message.success("知识库已创建");
    },
    onError: (error) => message.error(readError(error)),
  });
  const updateKb = useMutation({
    mutationFn: ({ id, values }) => api.patch(`/knowledge-bases/${id}`, values),
    onSuccess: () => {
      setEditingKb(null);
      queryClient.invalidateQueries({ queryKey: ["kbs"] });
      message.success("知识库已更新");
    },
    onError: (error) => message.error(readError(error)),
  });
  const deleteKb = useMutation({
    mutationFn: (id) => api.delete(`/knowledge-bases/${id}`),
    onSuccess: () => {
      setSelectedKb("");
      queryClient.invalidateQueries({ queryKey: ["kbs"] });
      message.success("知识库已删除");
    },
    onError: (error) => message.error(readError(error)),
  });
  const ingestUrl = useMutation({
    mutationFn: (values) => api.post(`/knowledge-bases/${activeKb}/urls/plan`, {
      urls: splitUrls(values.urls),
      skip_duplicates: values.skip_duplicates !== false,
    }),
    onSuccess: ({ data }) => {
      setUrlPlan(data);
      setSelectedUrlItemIds((data.items || []).filter((item) => item.can_enqueue && item.severity === "pass").map((item) => item.client_item_id));
      message.success(`校验完成：可入队 ${data.ready_count}，阻断 ${data.blocked_count}`);
    },
    onError: (error) => message.error(readError(error)),
  });
  const commitUrlPlan = useMutation({
    mutationFn: () => api.post(`/knowledge-bases/${activeKb}/urls/batch`, {
      plan_id: urlPlan.plan_id,
      client_item_ids: selectedUrlItemIds,
    }),
    onSuccess: ({ data }) => {
      urlForm.resetFields();
      setUrlPlan(null);
      setSelectedUrlItemIds([]);
      refreshKnowledge(queryClient, activeKb);
      message.success(`确认入队完成：成功 ${data.succeeded}，失败 ${data.failed}，跳过 ${data.skipped || 0}`);
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
  const retryJob = useMutation({
    mutationFn: (jobId) => api.post(`/ingest-jobs/${jobId}/retry`, {}),
    onSuccess: () => {
      refreshKnowledge(queryClient, activeKb);
      message.success("任务已重新入队");
    },
    onError: (error) => message.error(readError(error)),
  });
  const cancelJob = useMutation({
    mutationFn: (jobId) => api.post(`/ingest-jobs/${jobId}/cancel`, {}),
    onSuccess: () => {
      refreshKnowledge(queryClient, activeKb);
      message.success("任务已取消");
    },
    onError: (error) => message.error(readError(error)),
  });
  const batchDeleteDocs = useMutation({
    mutationFn: (fileIds) => api.post(`/knowledge-bases/${activeKb}/documents/batch-delete`, { file_ids: fileIds }),
    onSuccess: ({ data }) => {
      setSelectedFileIds([]);
      refreshKnowledge(queryClient, activeKb);
      message.success(`批量删除完成：成功 ${data.succeeded}，失败 ${data.failed}`);
    },
    onError: (error) => message.error(readError(error)),
  });
  const batchReindexDocs = useMutation({
    mutationFn: (fileIds) => api.post(`/knowledge-bases/${activeKb}/documents/batch-reindex`, { file_ids: fileIds }),
    onSuccess: ({ data }) => {
      setSelectedFileIds([]);
      refreshKnowledge(queryClient, activeKb);
      message.success(`批量重建已入队：成功 ${data.succeeded}，失败 ${data.failed}`);
    },
    onError: (error) => message.error(readError(error)),
  });
  const batchRetryJobs = useMutation({
    mutationFn: (jobIds) => api.post("/ingest-jobs/actions/batch-retry", { job_ids: jobIds }),
    onSuccess: ({ data }) => {
      setSelectedJobIds([]);
      refreshKnowledge(queryClient, activeKb);
      message.success(`批量重试完成：成功 ${data.succeeded}，失败 ${data.failed}`);
    },
    onError: (error) => message.error(readError(error)),
  });
  const batchCancelJobs = useMutation({
    mutationFn: (jobIds) => api.post("/ingest-jobs/actions/batch-cancel", { job_ids: jobIds }),
    onSuccess: ({ data }) => {
      setSelectedJobIds([]);
      refreshKnowledge(queryClient, activeKb);
      message.success(`批量取消完成：成功 ${data.succeeded}，失败 ${data.failed}`);
    },
    onError: (error) => message.error(readError(error)),
  });
  const upsertGrant = useMutation({
    mutationFn: (values) => api.put(`/knowledge-bases/${activeKb}/grants`, values),
    onSuccess: () => {
      memberForm.resetFields();
      queryClient.invalidateQueries({ queryKey: ["kb-grants", activeKb] });
      queryClient.invalidateQueries({ queryKey: ["kbs"] });
      message.success("授权已保存");
    },
    onError: (error) => message.error(readError(error)),
  });
  const removeGrant = useMutation({
    mutationFn: (grantId) => api.delete(`/knowledge-bases/${activeKb}/grants/${grantId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["kb-grants", activeKb] });
      queryClient.invalidateQueries({ queryKey: ["kbs"] });
      message.success("授权已移除");
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

  function openEditKb(item) {
    setEditingKb(item);
    editForm.setFieldsValue({
      name: item.name,
      description: item.description,
      visibility: item.visibility,
      department_ids: item.department_ids || [],
      retrieval_top_k: item.retrieval_top_k,
      rerank_top_n: item.rerank_top_n,
      low_confidence_threshold: item.low_confidence_threshold ?? 0.35,
      low_confidence_max_retries: item.low_confidence_max_retries ?? 1,
    });
  }

  function confirmDeleteKb(item) {
    Modal.confirm({
      title: `删除知识库「${item.name}」？`,
      content: "删除后该知识库、文件、任务和 API Key 会从工作台隐藏，数据采用软删除。",
      okText: "删除",
      okButtonProps: { danger: true },
      cancelText: "取消",
      onOk: () => deleteKb.mutate(item.id),
    });
  }

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
          <Form.Item name="department_ids" label="授权部门">
            <Select mode="multiple" allowClear options={(departments.data || []).map((item) => ({ value: item.id, label: item.name }))} />
          </Form.Item>
          <Flex gap={8} wrap>
            <Form.Item name="retrieval_top_k" label="Top K" className="grow">
              <InputNumber min={1} max={50} placeholder="全局默认" className="full-select" />
            </Form.Item>
            <Form.Item name="rerank_top_n" label="重排数" className="grow">
              <InputNumber min={0} max={50} placeholder="全局默认" className="full-select" />
            </Form.Item>
          </Flex>
          <Flex gap={8} wrap>
            <Form.Item name="low_confidence_threshold" label="低可信阈值" initialValue={0.35} className="grow">
              <InputNumber min={0} max={1} step={0.05} className="full-select" />
            </Form.Item>
            <Form.Item name="low_confidence_max_retries" label="最大重试" initialValue={1} className="grow">
              <InputNumber min={0} max={3} className="full-select" />
            </Form.Item>
          </Flex>
          <Button type="primary" htmlType="submit" loading={createKb.isPending}>创建知识库</Button>
        </Form>
        <div className="kb-list">
          {fullAccessKbs.map((item) => (
            <div
              className={activeKb === item.id ? "kb-item active" : "kb-item"}
              key={item.id}
              onClick={() => setSelectedKb(item.id)}
              onKeyDown={(event) => activateOnKeyboard(event, () => setSelectedKb(item.id))}
              role="button"
              tabIndex={0}
            >
              <Flex justify="space-between" align="start" gap={8} wrap>
                <div>
                  <strong>{item.name}</strong>
                  <span>{item.visibility} · TopK {item.retrieval_top_k || "全局"} · 重试 {item.low_confidence_max_retries ?? 1} · {item.completed_file_count}/{item.file_count} 文件</span>
                </div>
                <Space onClick={(event) => event.stopPropagation()}>
              {item.has_full_access && <Button size="small" icon={<EditOutlined />} onClick={() => openEditKb(item)} />}
              {item.has_full_access && <Button size="small" danger icon={<DeleteOutlined />} onClick={() => confirmDeleteKb(item)} />}
                </Space>
              </Flex>
            </div>
          ))}
        </div>
      </div>
      <div className="surface main">
        <Flex justify="space-between" align="center" gap={12} wrap>
          <PageTitle
            title={activeKbRecord ? `${activeKbRecord.name} · 文档管理` : "文档管理"}
            subtitle={activeKbRecord?.description || "支持文本、Markdown、HTML、Word、PDF 解析，PDF 可回看页文本和块定位。"}
          />
          <Button icon={<ReloadOutlined />} onClick={() => refreshKnowledge(queryClient, activeKb)}>刷新</Button>
        </Flex>
        {activeKbRecord && (
          <div className="stats-row">
            <Tag color="blue">文件 {activeKbRecord.file_count}</Tag>
            <Tag color="green">已完成 {activeKbRecord.completed_file_count}</Tag>
            <Tag color={activeKbRecord.failed_job_count ? "red" : "default"}>失败任务 {activeKbRecord.failed_job_count}</Tag>
            <Tag>{activeKbRecord.visibility}</Tag>
            <Tag color={activeKbRecord.can_write ? "blue" : "default"}>{activeKbRecord.current_user_role}</Tag>
          </div>
        )}
        {queueHealth.data && (
          <div className="queue-health">
            <div><strong>{queueHealth.data.redis_queue_length}</strong><span>Redis 队列</span></div>
            <div><strong>{queueHealth.data.pending_count}</strong><span>等待</span></div>
            <div><strong>{queueHealth.data.running_count}</strong><span>运行</span></div>
            <div><strong>{queueHealth.data.failed_count}</strong><span>失败</span></div>
            <div><strong>{queueHealth.data.oldest_pending_wait_seconds == null ? "-" : `${queueHealth.data.oldest_pending_wait_seconds}s`}</strong><span>最久等待</span></div>
            <div><strong className={queueHealth.data.worker_stale ? "danger-text" : ""}>{queueHealth.data.worker_stale ? "异常" : "在线"}</strong><span>Worker</span></div>
          </div>
        )}
        <Upload.Dragger {...uploadProps} disabled={!activeKb || !canWriteKb} className="upload-zone">
          <p><CloudUploadOutlined /></p>
          <p>拖拽或点击上传文档</p>
        </Upload.Dragger>
        <Form form={urlForm} className="url-form stacked-form" layout="vertical" onFinish={(values) => ingestUrl.mutate(values)}>
          <Form.Item name="urls" rules={[{ required: true }]} className="grow">
            <Input.TextArea rows={4} disabled={!canWriteKb} placeholder="每行一个 URL，先校验再确认入队" />
          </Form.Item>
          <Space wrap>
            <Form.Item name="skip_duplicates" initialValue={true} valuePropName="checked" noStyle>
              <Switch checkedChildren="跳过重复" unCheckedChildren="允许重复" />
            </Form.Item>
            <Button type="primary" htmlType="submit" disabled={!canWriteKb} loading={ingestUrl.isPending}>开始校验</Button>
            <Button disabled={!canWriteKb || !urlPlan || !selectedUrlItemIds.length} loading={commitUrlPlan.isPending} onClick={() => commitUrlPlan.mutate()}>确认入队</Button>
          </Space>
        </Form>
        {urlPlan && (
          <div className="import-plan">
            <Flex justify="space-between" align="center" className="table-tools" wrap>
              <Space wrap>
                <Tag color="blue">总数 {urlPlan.total}</Tag>
                <Tag color="green">可入队 {urlPlan.ready_count}</Tag>
                <Tag color="orange">警告 {urlPlan.warning_count}</Tag>
                <Tag color="red">阻断 {urlPlan.blocked_count}</Tag>
              </Space>
              <span className="muted">计划过期：{formatTime(urlPlan.expires_at)}</span>
            </Flex>
            <Table
              rowKey="client_item_id"
              size="small"
              pagination={{ pageSize: 5 }}
              dataSource={urlPlan.items || []}
              rowSelection={{
                selectedRowKeys: selectedUrlItemIds,
                onChange: setSelectedUrlItemIds,
                getCheckboxProps: (row) => ({ disabled: !row.can_enqueue }),
              }}
              columns={[
                { title: "URL", dataIndex: "url", ellipsis: true },
                { title: "状态", dataIndex: "status", width: 160, render: (value, row) => <Tag color={planStatusColor(row)}>{value}</Tag> },
                { title: "原因", dataIndex: "reason", ellipsis: true, render: (value) => value || "-" },
                { title: "分块", dataIndex: "estimated_chunks", width: 80, render: (value) => value ?? "-" },
                { title: "重复", width: 120, render: (_, row) => row.duplicate_file_id?.slice(0, 8) || row.duplicate_of || "-" },
              ]}
            />
          </div>
        )}
        <Flex justify="space-between" align="center" className="table-tools" wrap>
          <Typography.Title level={5}>文档处理任务中心</Typography.Title>
          <Space wrap>
            <Select
              value={taskStatus}
              onChange={setTaskStatus}
              style={{ width: 160 }}
              options={[
                { value: "all", label: "全部任务" },
                { value: "pending", label: "等待中" },
                { value: "processing", label: "处理中" },
                { value: "completed", label: "已完成" },
                { value: "failed", label: "失败" },
                { value: "cancelled", label: "已取消" },
              ]}
            />
          </Space>
        </Flex>
        <Table
          rowKey="id"
          size="middle"
          loading={documentTasks.isLoading}
          dataSource={documentTasks.data || []}
          scroll={{ x: "max-content" }}
          columns={[
            { title: "文件/来源", render: (_, row) => <Space direction="vertical" size={0}><strong>{row.filename || row.source_uri || row.id}</strong><span className="muted">{row.content_type || row.source_type || "-"} · {formatBytes(row.file_size)}</span></Space> },
            { title: "状态", width: 140, render: (_, row) => <Space direction="vertical" size={0}><Tag color={taskStatusColor(row.status)}>{taskStatusText(row.status)}</Tag>{row.is_stale && <span className="danger-text">等待超时 {row.stale_seconds}s</span>}</Space> },
            { title: "阶段", dataIndex: "stage", width: 120, render: taskStageText },
            { title: "进度", dataIndex: "progress", width: 170, render: (value) => <Progress percent={value} size="small" status={value >= 100 ? undefined : "active"} /> },
            { title: "分块/向量", width: 110, render: (_, row) => `${row.chunk_count || 0}/${row.vector_count || 0}` },
            { title: "上传人", dataIndex: "uploaded_by_user_id", width: 110, render: (value) => value?.slice(0, 8) || "-" },
            { title: "更新时间", dataIndex: "updated_at", width: 180, render: formatTime },
            { title: "错误", dataIndex: "error_message", ellipsis: true, render: (value) => value || "-" },
            {
              title: "操作",
              width: 280,
              render: (_, row) => (
                <Space>
                  <Button size="small" icon={<EyeOutlined />} onClick={() => setTaskDetail(row)}>详情</Button>
                  {row.can_preview && <Button size="small" onClick={() => setPreview({ id: row.file_id, filename: row.filename })}>预览</Button>}
                  {row.can_retry && <Button size="small" onClick={() => retryJob.mutate(row.job_id)}>重试</Button>}
                  {row.can_cancel && <Button size="small" danger onClick={() => cancelJob.mutate(row.job_id)}>取消</Button>}
                  {row.can_delete && <Button size="small" danger icon={<DeleteOutlined />} onClick={() => deleteDoc.mutate(row.file_id)} />}
                </Space>
              ),
            },
          ]}
        />
        <Flex justify="space-between" align="center" className="table-tools" wrap>
          <Typography.Title level={5}>文件</Typography.Title>
          <Space wrap>
            <Button disabled={!canManageKb || !selectedFileIds.length} icon={<ReloadOutlined />} onClick={() => batchReindexDocs.mutate(selectedFileIds)}>批量重建</Button>
            <Button disabled={!canManageKb || !selectedFileIds.length} danger icon={<DeleteOutlined />} onClick={() => batchDeleteDocs.mutate(selectedFileIds)}>批量删除</Button>
            <Select
              value={fileStatus}
              onChange={setFileStatus}
              style={{ width: 160 }}
              options={[
                { value: "", label: "全部文件" },
                { value: "processing", label: "处理中" },
                { value: "completed", label: "已完成" },
                { value: "failed", label: "失败" },
                { value: "deleted", label: "已删除" },
              ]}
            />
          </Space>
        </Flex>
        <Table
          rowKey="id"
          size="middle"
          loading={files.isLoading}
          dataSource={files.data || []}
          scroll={{ x: "max-content" }}
          rowSelection={canManageKb ? { selectedRowKeys: selectedFileIds, onChange: setSelectedFileIds } : undefined}
          columns={[
            { title: "文件", dataIndex: "filename", render: (value, row) => <Space direction="vertical" size={0}><strong>{value}</strong><span className="muted">{row.content_type}</span></Space> },
            { title: "状态", dataIndex: "status", width: 120, render: (value) => <Tag color={value === "completed" ? "green" : value === "failed" ? "red" : "blue"}>{value}</Tag> },
            { title: "分块", dataIndex: "chunk_count", width: 90 },
            { title: "错误", dataIndex: "error_message", ellipsis: true },
            { title: "更新时间", dataIndex: "updated_at", width: 190, render: formatTime },
            {
              title: "操作",
              width: 230,
              render: (_, row) => (
                <Space>
                  <Button icon={<EyeOutlined />} onClick={() => setPreview(row)}>预览</Button>
                  {canManageKb && <Button icon={<ReloadOutlined />} onClick={() => reindexDoc.mutate(row.id)}>重建</Button>}
                  {canManageKb && <Button danger icon={<DeleteOutlined />} onClick={() => deleteDoc.mutate(row.id)} />}
                </Space>
              ),
            },
          ]}
        />
        <Typography.Title level={5}>入库任务</Typography.Title>
        <Space className="table-tools" wrap>
          <Button disabled={!canManageKb || !selectedJobIds.length} onClick={() => batchRetryJobs.mutate(selectedJobIds)}>批量重试失败</Button>
          <Button disabled={!canManageKb || !selectedJobIds.length} danger onClick={() => batchCancelJobs.mutate(selectedJobIds)}>批量取消</Button>
        </Space>
        <Tabs
          items={[
            {
              key: "active",
              label: `处理中 ${activeJobs.data?.length || 0}`,
              children: <JobTable jobs={activeJobs.data || []} loading={activeJobs.isLoading} retryJob={retryJob} cancelJob={cancelJob} selectedJobIds={selectedJobIds} setSelectedJobIds={setSelectedJobIds} canWrite={canManageKb} />,
            },
            {
              key: "history",
              label: <span><HistoryOutlined /> 历史 {historyJobs.data?.length || 0}</span>,
              children: <JobTable jobs={historyJobs.data || []} loading={historyJobs.isLoading} retryJob={retryJob} cancelJob={cancelJob} selectedJobIds={selectedJobIds} setSelectedJobIds={setSelectedJobIds} canWrite={canManageKb} />,
            },
          ]}
        />
        <Typography.Title level={5}>成员与部门权限</Typography.Title>
        {canManageMembers && (
          <Form
            form={memberForm}
            layout="inline"
            className="url-form"
            initialValues={{ subject_type: "user", role: "viewer" }}
            onFinish={(values) => upsertGrant.mutate(values)}
          >
            <Form.Item name="subject_type" rules={[{ required: true }]}>
              <Select
                style={{ width: 120 }}
                options={[
                  { value: "user", label: "用户" },
                  { value: "department", label: "部门" },
                ]}
                onChange={() => memberForm.resetFields(["subject_id"])}
              />
            </Form.Item>
            <Form.Item name="subject_id" rules={[{ required: true }]} className="kb-select">
              <Select
                showSearch
                placeholder={grantSubjectType === "department" ? "选择部门" : "选择用户"}
                optionFilterProp="label"
                options={
                  grantSubjectType === "department"
                    ? (departments.data || []).map((item) => ({ value: item.id, label: item.name }))
                    : (memberCandidates.data || []).map((item) => ({ value: item.id, label: `${item.display_name} · ${item.email}` }))
                }
              />
            </Form.Item>
            <Form.Item name="role" rules={[{ required: true }]}>
              <Select style={{ width: 140 }} options={[
                { value: "viewer", label: "viewer" },
                { value: "editor", label: "editor" },
                { value: "admin", label: "admin" },
              ]} />
            </Form.Item>
            <Button type="primary" htmlType="submit" loading={upsertGrant.isPending}>授权</Button>
          </Form>
        )}
        <Table
          rowKey="id"
          size="small"
          loading={grants.isLoading}
          pagination={false}
          dataSource={grants.data || []}
          columns={[
            {
              title: "授权对象",
              render: (_, row) => (
                <Space direction="vertical" size={0}>
                  <strong>{row.subject_name || row.subject_id}</strong>
                  <span className="muted">
                    {row.subject_type === "department" ? "部门" : row.subject_email || "用户"}
                  </span>
                </Space>
              ),
            },
            {
              title: "类型",
              dataIndex: "subject_type",
              width: 90,
              render: (value) => <Tag icon={value === "department" ? <BankOutlined /> : <TeamOutlined />}>{value === "department" ? "部门" : "用户"}</Tag>,
            },
            { title: "角色", dataIndex: "role", width: 120, render: (value) => <Tag color={value === "admin" ? "purple" : value === "editor" ? "blue" : "default"}>{value}</Tag> },
            { title: "授权人", dataIndex: "granted_by_user_id", width: 110, render: (value) => value?.slice(0, 8) || "-" },
            { title: "更新时间", dataIndex: "updated_at", width: 180, render: formatTime },
            {
              title: "操作",
              width: 90,
              render: (_, row) => canManageMembers ? <Button size="small" danger onClick={() => removeGrant.mutate(row.id)}>移除</Button> : "-",
            },
          ]}
        />
      </div>
      <DocumentPreview api={api} kbId={activeKb} file={preview} onClose={() => setPreview(null)} />
      <Modal title="文档处理详情" open={Boolean(taskDetail)} footer={<Button onClick={() => setTaskDetail(null)}>关闭</Button>} onCancel={() => setTaskDetail(null)} width={760}>
        {taskDetail && (
          <Descriptions
            bordered
            column={1}
            size="small"
            items={[
              { key: "id", label: "任务 ID", children: taskDetail.id },
              { key: "file", label: "文件 ID", children: taskDetail.file_id || "-" },
              { key: "job", label: "入库任务 ID", children: taskDetail.job_id || "-" },
              { key: "status", label: "状态", children: `${taskStatusText(taskDetail.status)} · ${taskStageText(taskDetail.stage)} · ${taskDetail.progress}%` },
              { key: "source", label: "来源", children: taskDetail.source_uri || taskDetail.filename || "-" },
              { key: "counts", label: "分块/向量", children: `${taskDetail.chunk_count || 0}/${taskDetail.vector_count || 0}` },
              { key: "retry", label: "重试次数", children: taskDetail.retry_count || 0 },
              { key: "created", label: "创建时间", children: formatTime(taskDetail.created_at) },
              { key: "started", label: "开始时间", children: formatTime(taskDetail.started_at) },
              { key: "completed", label: "完成时间", children: formatTime(taskDetail.completed_at) },
              { key: "updated", label: "更新时间", children: formatTime(taskDetail.updated_at) },
              { key: "error", label: "错误信息", children: taskDetail.error_message || "-" },
            ]}
          />
        )}
      </Modal>
      <Modal
        title="编辑知识库"
        open={Boolean(editingKb)}
        onCancel={() => setEditingKb(null)}
        okText="保存"
        confirmLoading={updateKb.isPending}
        onOk={() => editForm.validateFields().then((values) => updateKb.mutate({ id: editingKb.id, values }))}
      >
        <Form form={editForm} layout="vertical">
          <Form.Item name="name" label="名称" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input.TextArea rows={3} />
          </Form.Item>
          <Form.Item name="visibility" label="可见范围" rules={[{ required: true }]}>
            <Select options={[{ value: "department", label: "部门" }, { value: "org", label: "全组织" }, { value: "private", label: "私有" }]} />
          </Form.Item>
          <Form.Item name="department_ids" label="授权部门">
            <Select mode="multiple" allowClear options={(departments.data || []).map((item) => ({ value: item.id, label: item.name }))} />
          </Form.Item>
          <Flex gap={8} wrap>
            <Form.Item name="retrieval_top_k" label="Top K" className="grow">
              <InputNumber min={1} max={50} placeholder="全局默认" className="full-select" />
            </Form.Item>
            <Form.Item name="rerank_top_n" label="重排数" className="grow">
              <InputNumber min={0} max={50} placeholder="全局默认" className="full-select" />
            </Form.Item>
          </Flex>
          <Flex gap={8} wrap>
            <Form.Item name="low_confidence_threshold" label="低可信阈值" className="grow">
              <InputNumber min={0} max={1} step={0.05} className="full-select" />
            </Form.Item>
            <Form.Item name="low_confidence_max_retries" label="最大重试" className="grow">
              <InputNumber min={0} max={3} className="full-select" />
            </Form.Item>
          </Flex>
        </Form>
      </Modal>
    </section>
  );
}

function JobTable({ jobs, loading, retryJob, cancelJob, selectedJobIds, setSelectedJobIds, canWrite }) {
  return (
    <Table
      rowKey="id"
      size="small"
      loading={loading}
      pagination={{ pageSize: 6 }}
      dataSource={jobs}
      scroll={{ x: "max-content" }}
      rowSelection={canWrite ? { selectedRowKeys: selectedJobIds, onChange: setSelectedJobIds } : undefined}
      columns={[
        { title: "来源", render: (_, row) => row.filename || row.source_uri },
        { title: "状态", dataIndex: "status", width: 120, render: (value) => <Tag color={value === "failed" ? "red" : value === "succeeded" ? "green" : "blue"}>{value}</Tag> },
        { title: "进度", dataIndex: "progress", width: 170, render: (value) => <Progress percent={value} size="small" /> },
        { title: "耗时", dataIndex: "duration_ms", width: 100, render: (value) => value == null ? "-" : `${Math.round(value / 1000)}s` },
        { title: "更新时间", dataIndex: "updated_at", width: 180, render: formatTime },
        { title: "错误", dataIndex: "error_message", ellipsis: true },
        {
          title: "操作",
          width: 150,
          render: (_, row) => (
            <Space>
              {canWrite && row.status === "failed" && <Button size="small" onClick={() => retryJob.mutate(row.id)}>重试</Button>}
              {canWrite && ["pending", "running"].includes(row.status) && <Button size="small" danger onClick={() => cancelJob.mutate(row.id)}>取消</Button>}
            </Space>
          ),
        },
      ]}
    />
  );
}

function SourceDrawer({ source, onClose, onOpenPreview }) {
  return (
    <Drawer rootClassName="source-drawer" width={520} title={source?.filename || "引用来源"} open={Boolean(source)} onClose={onClose}>
      {source && (
        <Space direction="vertical" size={14} className="full-select">
          <Space wrap>
            <Tag color="blue">[{source.source_index || "-"}]</Tag>
            <Tag>{source.source_type || "document"}</Tag>
            <Tag>分块 {source.chunk_index ?? "-"}</Tag>
            <Tag>页码 {source.page_number ?? "-"}</Tag>
          </Space>
          <Descriptions
            bordered
            column={1}
            size="small"
            items={[
              { key: "filename", label: "文件", children: source.filename || "-" },
              { key: "file_id", label: "文件 ID", children: source.file_id || "-" },
              { key: "chunk_id", label: "分块 ID", children: source.chunk_id || "-" },
              { key: "source_uri", label: "来源 URI", children: source.source_uri || "-" },
              { key: "score", label: "相关分", children: scoreText(source.rerank_score ?? source.hybrid_score ?? source.vector_score) },
            ]}
          />
          <Typography.Title level={5}>原文片段</Typography.Title>
          <div className="source-snippet">{source.snippet || "-"}</div>
          <Button icon={<EyeOutlined />} disabled={!source.file_id} onClick={() => onOpenPreview(source)}>打开文档预览</Button>
        </Space>
      )}
    </Drawer>
  );
}

function DocumentPreview({ api, kbId, file, onClose }) {
  const preview = useQuery({
    queryKey: ["preview", kbId, file?.id],
    enabled: Boolean(kbId && file?.id),
    queryFn: async () => (await api.get(`/knowledge-bases/${kbId}/documents/${file.id}/preview`)).data,
  });
  return (
    <Drawer rootClassName="wide-drawer" width="72vw" title={file?.filename} open={Boolean(file)} onClose={onClose}>
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
                      <Flex justify="space-between" wrap>
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
  const queryClient = useQueryClient();
  const [kbId, setKbId] = useState("");
  const [conversationId, setConversationId] = useState("");
  const [messages, setMessages] = useState([]);
  const [activeSource, setActiveSource] = useState(null);
  const [previewSourceFile, setPreviewSourceFile] = useState(null);
  const [form] = Form.useForm();
  const kbs = useQuery({ queryKey: ["kbs"], queryFn: async () => (await api.get("/knowledge-bases")).data.items || [] });
  const activeKbRecord = (kbs.data || []).find((item) => item.id === kbId);
  const canAsk = Boolean(kbId && (activeKbRecord?.completed_file_count || 0) > 0);
  const conversations = useQuery({
    queryKey: ["conversations", kbId],
    enabled: Boolean(kbId),
    queryFn: async () => (await api.get("/conversations", { params: { knowledge_base_id: kbId } })).data.items || [],
  });
  useEffect(() => {
    if (!kbId && kbs.data?.[0]?.id) setKbId(kbs.data[0].id);
  }, [kbs.data, kbId]);
  const ask = useMutation({
    mutationFn: (values) => api.post("/chat", { knowledge_base_id: kbId, conversation_id: conversationId || undefined, message: values.message }),
    onSuccess: ({ data }, values) => {
      setConversationId(data.conversation_id);
      setMessages((items) => [{ question: values.message, ...data }, ...items]);
      form.resetFields();
      queryClient.invalidateQueries({ queryKey: ["conversations", kbId] });
    },
    onError: (error) => message.error(readError(error)),
  });
  const renameConversation = useMutation({
    mutationFn: ({ id, title }) => api.patch(`/conversations/${id}`, { title }, { params: { knowledge_base_id: kbId } }),
    onSuccess: () => {
      message.success("会话已重命名");
      queryClient.invalidateQueries({ queryKey: ["conversations", kbId] });
    },
    onError: (error) => message.error(readError(error)),
  });
  const deleteConversationMutation = useMutation({
    mutationFn: (id) => api.delete(`/conversations/${id}`, { params: { knowledge_base_id: kbId } }),
    onSuccess: (_, id) => {
      if (conversationId === id) startNewConversation();
      message.success("会话已删除");
      queryClient.invalidateQueries({ queryKey: ["conversations", kbId] });
    },
    onError: (error) => message.error(readError(error)),
  });
  const feedbackMutation = useMutation({
    mutationFn: (payload) => api.post("/feedback", payload),
    onSuccess: () => message.success("反馈已记录"),
    onError: (error) => message.error(readError(error)),
  });

  async function openConversation(item) {
    try {
      const { data } = await api.get(`/conversations/${item.id}/messages`, { params: { knowledge_base_id: kbId } });
      setConversationId(item.id);
      setMessages(buildChatRecordsFromMessages(data.items || []));
    } catch (error) {
      setConversationId("");
      setMessages([]);
      message.error(readError(error));
    }
  }

  function startNewConversation() {
    setConversationId("");
    setMessages([]);
    form.resetFields();
  }

  function renameItem(event, item) {
    event.stopPropagation();
    let nextTitle = item.title || "";
    Modal.confirm({
      title: "重命名会话",
      content: <Input defaultValue={nextTitle} maxLength={120} onChange={(inputEvent) => { nextTitle = inputEvent.target.value; }} />,
      okText: "保存",
      cancelText: "取消",
      onOk: () => {
        const title = nextTitle.trim();
        if (!title) {
          message.warning("请输入会话名称");
          return Promise.reject(new Error("empty title"));
        }
        return renameConversation.mutateAsync({ id: item.id, title });
      },
    });
  }

  function deleteItem(event, item) {
    event.stopPropagation();
    Modal.confirm({
      title: "删除会话",
      content: `确认删除“${item.title || "未命名会话"}”？该会话消息也会一起删除。`,
      okText: "删除",
      okButtonProps: { danger: true },
      cancelText: "取消",
      onOk: () => deleteConversationMutation.mutateAsync(item.id),
    });
  }

  function submitFeedback(item, rating, extra = {}) {
    feedbackMutation.mutate({
      knowledge_base_id: kbId,
      conversation_id: item.conversation_id || conversationId,
      assistant_message_id: item.assistant_message_id,
      rating,
      question: item.question,
      answer: item.answer,
      reason: extra.reason,
      comment: extra.comment,
      sources_snapshot: item.sources || [],
    });
  }

  function openSource(source) {
    setActiveSource(source);
  }

  function openSourcePreview(source) {
    if (!source?.file_id) return;
    setPreviewSourceFile({ id: source.file_id, filename: source.filename || "来源文档" });
  }

  function dislikeAnswer(item) {
    let reason = "answer_wrong";
    let comment = "";
    Modal.confirm({
      title: "反馈问题",
      content: (
        <Space direction="vertical" className="full-select">
          <Select
            defaultValue={reason}
            options={[
              { value: "answer_wrong", label: "答案不正确" },
              { value: "source_missing", label: "缺少引用" },
              { value: "source_mismatch", label: "引用不匹配" },
              { value: "not_answered", label: "没有回答问题" },
              { value: "content_outdated", label: "内容过时" },
              { value: "permission_leak", label: "权限不应可见" },
              { value: "unclear", label: "表达不清楚" },
              { value: "other", label: "其他" },
            ]}
            onChange={(value) => { reason = value; }}
          />
          <Input.TextArea rows={3} placeholder="补充说明" onChange={(event) => { comment = event.target.value; }} />
        </Space>
      ),
      okText: "提交",
      cancelText: "取消",
      onOk: () => submitFeedback(item, "down", { reason, comment }),
    });
  }

  return (
    <section className="page-grid chat-grid">
      <div className="surface narrow">
        <PageTitle title="会话历史" subtitle="按知识库隔离，只展示当前用户自己的会话。" />
        <Select
          className="full-select"
          value={kbId}
          onChange={(value) => { setKbId(value); startNewConversation(); }}
          options={(kbs.data || []).map((item) => ({ value: item.id, label: `${item.name} (${item.completed_file_count || 0})` }))}
        />
        <Button block className="new-chat-button" onClick={startNewConversation}>新建会话</Button>
        <div className="kb-list">
          {(conversations.data || []).map((item) => (
            <div
              className={conversationId === item.id ? "conversation-item active" : "conversation-item"}
              key={item.id}
              onClick={() => openConversation(item)}
              onKeyDown={(event) => activateOnKeyboard(event, () => openConversation(item))}
              role="button"
              tabIndex={0}
            >
              <div>
                <strong>{item.title || "未命名会话"}</strong>
                <span>{formatTime(item.updated_at)}</span>
              </div>
              <Space size={4}>
                <Button size="small" type="text" icon={<EditOutlined />} onClick={(event) => renameItem(event, item)} />
                <Button size="small" type="text" danger icon={<DeleteOutlined />} onClick={(event) => deleteItem(event, item)} />
              </Space>
            </div>
          ))}
        </div>
      </div>
      <div className="surface main">
        <PageTitle title="知识库问答" subtitle="多轮改写、混合检索、重排和引用来源会一起参与回答。" />
        {activeKbRecord && !canAsk && (
          <div className="empty-text compact">当前知识库还没有完成入库的文档，请先上传并等待处理完成。</div>
        )}
        <Form form={form} layout="inline" className="chat-form" onFinish={(values) => ask.mutate(values)}>
          <Form.Item name="message" rules={[{ required: true }]} className="grow">
            <Input.Search placeholder="输入问题" enterButton="发送" loading={ask.isPending} disabled={!canAsk} />
          </Form.Item>
        </Form>
        <div className="answer-list">
          {messages.map((item) => (
            <article className="answer-item" key={item.assistant_message_id || item.user_message_id}>
              <Flex justify="space-between" gap={12} align="start" wrap>
                <Typography.Title level={5}>{item.question}</Typography.Title>
                <Space>
                  <Tag color={confidenceColor(item.confidence)}>{confidenceText(item.confidence, item.confidence_score)}</Tag>
                  <Tag>{answerStatusText(item.answer_status)}</Tag>
                  {item.auto_retry_triggered && <Tag color="blue">自动重试 {item.retry_count}</Tag>}
                  {item.assistant_message_id && (
                    <>
                      <Button size="small" icon={<LikeOutlined />} onClick={() => submitFeedback(item, "up")} />
                      <Button size="small" icon={<DislikeOutlined />} onClick={() => dislikeAnswer(item)} />
                    </>
                  )}
                </Space>
              </Flex>
              <Typography.Paragraph className="answer-content">{renderAnswerWithCitations(item.answer, item.sources || [], openSource)}</Typography.Paragraph>
              <Table
                rowKey={(row, index) => `${row.chunk_id}-${index}`}
                size="small"
                pagination={false}
                dataSource={item.sources}
                columns={[
                  { title: "文件", dataIndex: "filename" },
                  { title: "引用", dataIndex: "source_index", width: 70, render: (value, row) => <Button size="small" type="link" onClick={() => openSource(row)}>[{value || "-"}]</Button> },
                  { title: "分块", dataIndex: "chunk_index", width: 80 },
                  { title: "页码", dataIndex: "page_number", width: 80, render: (value) => value || "-" },
                  { title: "分数", width: 140, render: (_, row) => scoreText(row.rerank_score ?? row.hybrid_score ?? row.vector_score) },
                  { title: "片段", dataIndex: "snippet" },
                ]}
              />
            </article>
          ))}
        </div>
      </div>
      <SourceDrawer source={activeSource} onClose={() => setActiveSource(null)} onOpenPreview={openSourcePreview} />
      <DocumentPreview api={api} kbId={kbId} file={previewSourceFile} onClose={() => setPreviewSourceFile(null)} />
    </section>
  );
}

function UsersWorkspace({ api, user }) {
  const queryClient = useQueryClient();
  const [deptForm] = Form.useForm();
  const [userForm] = Form.useForm();
  const [editForm] = Form.useForm();
  const [passwordForm] = Form.useForm();
  const [editingUser, setEditingUser] = useState(null);
  const [resettingUser, setResettingUser] = useState(null);
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
  const updateUserMutation = useMutation({
    mutationFn: ({ id, values }) => api.patch(`/users/${id}`, { ...values, department_id: values.department_id || null }),
    onSuccess: () => {
      setEditingUser(null);
      queryClient.invalidateQueries({ queryKey: ["users"] });
      message.success("用户已更新");
    },
    onError: (error) => message.error(readError(error)),
  });
  const resetPasswordMutation = useMutation({
    mutationFn: ({ id, password }) => api.post(`/users/${id}/reset-password`, { password }),
    onSuccess: () => {
      setResettingUser(null);
      passwordForm.resetFields();
      message.success("密码已重置");
    },
    onError: (error) => message.error(readError(error)),
  });
  const deactivateUserMutation = useMutation({
    mutationFn: (id) => api.delete(`/users/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["users"] });
      message.success("用户已禁用");
    },
    onError: (error) => message.error(readError(error)),
  });

  function openEditUser(row) {
    setEditingUser(row);
    editForm.setFieldsValue({
      display_name: row.display_name,
      role: row.role,
      department_id: row.department_id || undefined,
      is_active: row.is_active,
    });
  }

  function confirmDeactivate(row) {
    Modal.confirm({
      title: `禁用用户「${row.display_name}」？`,
      content: "禁用后该用户不能再登录，已有 token 下次校验会失效。",
      okText: "禁用",
      okButtonProps: { danger: true },
      cancelText: "取消",
      onOk: () => deactivateUserMutation.mutate(row.id),
    });
  }

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
          scroll={{ x: "max-content" }}
          columns={[
            { title: "用户", render: (_, row) => <Space direction="vertical" size={0}><strong>{row.display_name}</strong><span className="muted">{row.email}</span></Space> },
            { title: "角色", dataIndex: "role", render: (value) => <Tag color={value === "admin" ? "red" : value === "manager" ? "gold" : "blue"}>{value}</Tag> },
            { title: "状态", dataIndex: "is_active", render: (value) => <Tag color={value ? "green" : "red"}>{value ? "active" : "disabled"}</Tag> },
            { title: "部门", dataIndex: "department_id", render: (value) => departments.data?.find((item) => item.id === value)?.name || value?.slice(0, 8) || "-" },
            { title: "最近登录", dataIndex: "last_login_at", render: formatTime },
            { title: "创建时间", dataIndex: "created_at", render: formatTime },
            {
              title: "操作",
              width: 220,
              render: (_, row) => (
                <Space>
                  <Button size="small" icon={<EditOutlined />} onClick={() => openEditUser(row)}>编辑</Button>
                  <Button size="small" onClick={() => setResettingUser(row)}>重置密码</Button>
                  <Button size="small" danger disabled={!row.is_active || row.id === user?.id} onClick={() => confirmDeactivate(row)}>禁用</Button>
                </Space>
              ),
            },
          ]}
        />
      </div>
      <Modal
        title="编辑用户"
        open={Boolean(editingUser)}
        onCancel={() => setEditingUser(null)}
        okText="保存"
        confirmLoading={updateUserMutation.isPending}
        onOk={() => editForm.validateFields().then((values) => updateUserMutation.mutate({ id: editingUser.id, values }))}
      >
        <Form form={editForm} layout="vertical">
          <Form.Item name="display_name" label="姓名" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="role" label="角色" rules={[{ required: true }]}>
            <Select options={["member", "manager", "admin"].map((value) => ({ value, label: value }))} />
          </Form.Item>
          <Form.Item name="department_id" label="部门">
            <Select allowClear options={(departments.data || []).map((item) => ({ value: item.id, label: item.name }))} />
          </Form.Item>
          <Form.Item name="is_active" label="启用" valuePropName="checked">
            <Switch />
          </Form.Item>
        </Form>
      </Modal>
      <Modal
        title={`重置密码${resettingUser ? ` · ${resettingUser.display_name}` : ""}`}
        open={Boolean(resettingUser)}
        onCancel={() => setResettingUser(null)}
        okText="重置"
        confirmLoading={resetPasswordMutation.isPending}
        onOk={() => passwordForm.validateFields().then((values) => resetPasswordMutation.mutate({ id: resettingUser.id, password: values.password }))}
      >
        <Form form={passwordForm} layout="vertical">
          <Form.Item name="password" label="新密码" rules={[{ required: true, min: 8 }]}>
            <Input.Password />
          </Form.Item>
        </Form>
      </Modal>
    </section>
  );
}

function AuditWorkspace({ api, user }) {
  const queryClient = useQueryClient();
  const [chatFilters, setChatFilters] = useState({ feedback_rating: "", answer_status: "", knowledge_base_id: "", low_confidence: false, no_citations: false });
  const [qualityFilters, setQualityFilters] = useState({ knowledge_base_id: "", status: "", assignee_user_id: "" });
  const [chatOperation, setChatOperation] = useState(null);
  const [issueDraft, setIssueDraft] = useState(null);
  const [qualityIssue, setQualityIssue] = useState(null);
  const [issueForm] = Form.useForm();
  const [issueUpdateForm] = Form.useForm();
  const kbs = useQuery({ queryKey: ["kbs"], queryFn: async () => (await api.get("/knowledge-bases")).data.items || [] });
  const fullAccessKbs = (kbs.data || []).filter(isFullAccessKb);
  const canReadOperations = Boolean(user?.role === "admin" || fullAccessKbs.length);
  const logs = useQuery({ queryKey: ["audit"], enabled: canReadOperations, refetchInterval: 5000, queryFn: async () => (await api.get("/audit-logs")).data.items || [] });
  const feedback = useQuery({ queryKey: ["feedback"], refetchInterval: 10000, queryFn: async () => (await api.get("/feedback")).data.items || [] });
  const chatOperations = useQuery({
    queryKey: ["chat-operations", chatFilters],
    refetchInterval: 10000,
    queryFn: async () => (await api.get("/chat-operations", { params: compactParams(chatFilters) })).data.items || [],
  });
  const qualityIssues = useQuery({
    queryKey: ["quality-issues", qualityFilters],
    refetchInterval: 10000,
    queryFn: async () => (await api.get("/quality-issues", { params: compactParams(qualityFilters) })).data.items || [],
  });
  const assigneeKnowledgeBaseId = issueDraft?.knowledge_base_id || qualityIssue?.knowledge_base_id || qualityFilters.knowledge_base_id || "";
  const qualityIssueAssignees = useQuery({
    queryKey: ["quality-issue-assignees", assigneeKnowledgeBaseId],
    enabled: Boolean(assigneeKnowledgeBaseId),
    queryFn: async () => (await api.get("/quality-issue-assignees", { params: { knowledge_base_id: assigneeKnowledgeBaseId } })).data.items || [],
  });
  const chatOpsStats = useMemo(() => {
    const rows = chatOperations.data || [];
    const total = rows.length || 1;
    const low = rows.filter((item) => item.final_low_confidence || item.confidence === "low").length;
    const retried = rows.filter((item) => item.retry_count > 0).length;
    const improved = rows.filter((item) => item.retry_count > 0 && !item.final_low_confidence).length;
    return {
      lowRatio: `${Math.round((low / total) * 100)}%`,
      retried,
      improved,
      stillFailed: rows.filter((item) => item.retry_count > 0 && item.final_low_confidence).length,
    };
  }, [chatOperations.data]);
  const exportChatOperations = useMutation({
    mutationFn: async () => api.get("/chat-operations/export", { params: compactParams(chatFilters), responseType: "blob" }),
    onSuccess: ({ data }) => {
      const url = URL.createObjectURL(data);
      const link = document.createElement("a");
      link.href = url;
      link.download = "chat-operations.csv";
      link.click();
      URL.revokeObjectURL(url);
    },
    onError: (error) => message.error(readError(error)),
  });
  const createQualityIssueMutation = useMutation({
    mutationFn: (values) => api.post("/quality-issues", values),
    onSuccess: () => {
      setIssueDraft(null);
      issueForm.resetFields();
      queryClient.invalidateQueries({ queryKey: ["chat-operations"] });
      queryClient.invalidateQueries({ queryKey: ["quality-issues"] });
      message.success("质量待办已创建");
    },
    onError: (error) => message.error(readError(error)),
  });
  const updateQualityIssueMutation = useMutation({
    mutationFn: ({ id, values }) => api.patch(`/quality-issues/${id}`, values),
    onSuccess: ({ data }) => {
      setQualityIssue(data);
      queryClient.invalidateQueries({ queryKey: ["chat-operations"] });
      queryClient.invalidateQueries({ queryKey: ["quality-issues"] });
      message.success("质量待办已更新");
    },
    onError: (error) => message.error(readError(error)),
  });
  function updateChatFilter(key, value) {
    setChatFilters((filters) => ({ ...filters, [key]: value }));
  }
  function updateQualityFilter(key, value) {
    setQualityFilters((filters) => ({ ...filters, [key]: value }));
  }
  function openQualityIssueModal(row) {
    setIssueDraft(row);
    issueForm.setFieldsValue({
      knowledge_base_id: row.knowledge_base_id,
      assistant_message_id: row.assistant_message_id,
      issue_type: defaultIssueType(row),
      priority: row.final_low_confidence || row.feedback_rating === "down" ? "high" : "medium",
      assignee_user_id: undefined,
      resolution_note: row.feedback_comment || "",
    });
  }
  function submitQualityIssue() {
    issueForm.validateFields().then((values) => createQualityIssueMutation.mutate(values));
  }
  function openQualityIssue(row) {
    setQualityIssue(row);
    issueUpdateForm.setFieldsValue({
      status: row.status,
      priority: row.priority,
      assignee_user_id: row.assignee_user_id || undefined,
      resolution_note: row.resolution_note || "",
    });
  }
  function submitQualityIssueUpdate() {
    if (!qualityIssue) return;
    issueUpdateForm.validateFields().then((values) => updateQualityIssueMutation.mutate({ id: qualityIssue.id, values }));
  }
  const assigneeOptions = (qualityIssueAssignees.data || []).filter((item) => item.is_active).map((item) => ({ value: item.id, label: `${item.display_name} · ${item.email}` }));
  return (
    <section className="surface full">
      <PageTitle title="审计与反馈" subtitle="记录用户操作、请求上下文和回答质量反馈。" />
      <Tabs
        items={[
          {
            key: "chatOps",
            label: "问答运营",
            children: (
              <>
                <Space className="table-tools" wrap>
                  <Select
                    allowClear
                    placeholder="知识库"
                    value={chatFilters.knowledge_base_id || undefined}
                    style={{ width: 220 }}
                    onChange={(value) => updateChatFilter("knowledge_base_id", value || "")}
                    options={fullAccessKbs.map((item) => ({ value: item.id, label: item.name }))}
                  />
                  <Select
                    allowClear
                    placeholder="反馈"
                    value={chatFilters.feedback_rating || undefined}
                    style={{ width: 140 }}
                    onChange={(value) => updateChatFilter("feedback_rating", value || "")}
                    options={[
                      { value: "up", label: "有帮助" },
                      { value: "down", label: "无帮助" },
                      { value: "unrated", label: "未评价" },
                    ]}
                  />
                  <Select
                    allowClear
                    placeholder="答案状态"
                    value={chatFilters.answer_status || undefined}
                    style={{ width: 160 }}
                    onChange={(value) => updateChatFilter("answer_status", value || "")}
                    options={[
                      { value: "supported", label: "引用已验证" },
                      { value: "citation_missing", label: "缺少引用" },
                      { value: "citation_invalid", label: "引用无效" },
                      { value: "citation_incomplete", label: "引用不完整" },
                      { value: "no_sources", label: "无证据" },
                    ]}
                  />
                  <Switch checked={chatFilters.low_confidence} onChange={(value) => updateChatFilter("low_confidence", value)} checkedChildren="低可信" unCheckedChildren="低可信" />
                  <Switch checked={chatFilters.no_citations} onChange={(value) => updateChatFilter("no_citations", value)} checkedChildren="无引用" unCheckedChildren="无引用" />
                  <Button onClick={() => exportChatOperations.mutate()} loading={exportChatOperations.isPending}>导出 CSV</Button>
                </Space>
                <Space className="table-tools" wrap>
                  <Tag color="red">低可信 {chatOpsStats.lowRatio}</Tag>
                  <Tag color="blue">已重试 {chatOpsStats.retried}</Tag>
                  <Tag color="green">重试改善 {chatOpsStats.improved}</Tag>
                  <Tag color="orange">仍失败 {chatOpsStats.stillFailed}</Tag>
                </Space>
                <Table
                  rowKey="id"
                  loading={chatOperations.isLoading}
                  dataSource={chatOperations.data || []}
                  scroll={{ x: "max-content" }}
                  expandable={{ expandedRowRender: (row) => <ChatOperationDetail row={row} /> }}
                  columns={[
                    { title: "问题", dataIndex: "question", ellipsis: true },
                    { title: "状态", dataIndex: "answer_status", width: 130, render: answerStatusText },
                    { title: "反馈", dataIndex: "feedback_rating", width: 90, render: (value) => value ? <Tag color={value === "up" ? "green" : "red"}>{value === "up" ? "赞" : "踩"}</Tag> : "-" },
                    { title: "引用", width: 90, render: (_, row) => `${row.citation_count || 0}/${row.source_count || 0}` },
                    { title: "可信度", dataIndex: "confidence", width: 110, render: (value, row) => <Tag color={confidenceColor(value)}>{confidenceText(value, row.confidence_score)}</Tag> },
                    { title: "重试", dataIndex: "retry_count", width: 90, render: (value, row) => value ? <Tag color={row.final_low_confidence ? "red" : "blue"}>{value}</Tag> : "-" },
                    { title: "耗时", dataIndex: "latency_ms", width: 100, render: (value) => value == null ? "-" : `${value}ms` },
                    { title: "估算Tokens", dataIndex: "total_tokens", width: 110, render: (value) => value ?? "-" },
                    {
                      title: "质量待办",
                      width: 130,
                      render: (_, row) => row.quality_issue_id
                        ? <Tag color={qualityIssueStatusColor(row.quality_issue_status)}>{qualityIssueStatusText(row.quality_issue_status)}</Tag>
                        : isQualityIssueCandidate(row)
                        ? <Button size="small" onClick={() => openQualityIssueModal(row)}>创建待办</Button>
                        : "-",
                    },
                    { title: "时间", dataIndex: "created_at", width: 180, render: formatTime },
                    { title: "详情", width: 80, render: (_, row) => <Button size="small" onClick={() => setChatOperation(row)}>查看</Button> },
                  ]}
                />
                <Drawer rootClassName="wide-drawer" width="72vw" title="问答详情" open={Boolean(chatOperation)} onClose={() => setChatOperation(null)}>
                  {chatOperation && <ChatOperationDetail row={chatOperation} />}
                </Drawer>
              </>
            ),
          },
          {
            key: "quality",
            label: "质量待办",
            children: (
              <>
                <Space className="table-tools" wrap>
                  <Select
                    allowClear
                    placeholder="知识库"
                    value={qualityFilters.knowledge_base_id || undefined}
                    style={{ width: 220 }}
                    onChange={(value) => updateQualityFilter("knowledge_base_id", value || "")}
                    options={fullAccessKbs.map((item) => ({ value: item.id, label: item.name }))}
                  />
                  <Select
                    allowClear
                    placeholder="状态"
                    value={qualityFilters.status || undefined}
                    style={{ width: 150 }}
                    onChange={(value) => updateQualityFilter("status", value || "")}
                    options={qualityIssueStatusOptions()}
                  />
                  <Select
                    allowClear
                    showSearch
                    placeholder="负责人"
                    disabled={!qualityFilters.knowledge_base_id}
                    value={qualityFilters.assignee_user_id || undefined}
                    style={{ width: 260 }}
                    optionFilterProp="label"
                    onChange={(value) => updateQualityFilter("assignee_user_id", value || "")}
                    options={assigneeOptions}
                  />
                </Space>
                <Table
                  rowKey="id"
                  loading={qualityIssues.isLoading}
                  dataSource={qualityIssues.data || []}
                  scroll={{ x: "max-content" }}
                  expandable={{ expandedRowRender: (row) => <QualityIssueDetail row={row} /> }}
                  columns={[
                    { title: "问题", dataIndex: "question", ellipsis: true },
                    { title: "类型", dataIndex: "issue_type", width: 130, render: qualityIssueTypeText },
                    { title: "优先级", dataIndex: "priority", width: 100, render: (value) => <Tag color={qualityIssuePriorityColor(value)}>{qualityIssuePriorityText(value)}</Tag> },
                    { title: "状态", dataIndex: "status", width: 120, render: (value) => <Tag color={qualityIssueStatusColor(value)}>{qualityIssueStatusText(value)}</Tag> },
                    { title: "负责人", dataIndex: "assignee_user_id", width: 110, render: (value) => value?.slice(0, 8) || "-" },
                    { title: "反馈", dataIndex: "feedback_rating", width: 90, render: (value) => value ? <Tag color={value === "down" ? "red" : "green"}>{value === "down" ? "踩" : "赞"}</Tag> : "-" },
                    { title: "时间", dataIndex: "created_at", width: 180, render: formatTime },
                    { title: "操作", width: 90, render: (_, row) => <Button size="small" onClick={() => openQualityIssue(row)}>处理</Button> },
                  ]}
                />
              </>
            ),
          },
          {
            key: "audit",
            label: "审计日志",
            children: (
              <Table
                rowKey="id"
                loading={logs.isLoading}
                dataSource={logs.data || []}
                scroll={{ x: "max-content" }}
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
            ),
          },
          {
            key: "feedback",
            label: "回答反馈",
            children: (
              <Table
                rowKey="id"
                loading={feedback.isLoading}
                dataSource={feedback.data || []}
                scroll={{ x: "max-content" }}
                expandable={{ expandedRowRender: (row) => <pre className="json-block">{JSON.stringify(row, null, 2)}</pre> }}
                columns={[
                  { title: "评价", dataIndex: "rating", width: 90, render: (value) => <Tag color={value === "up" ? "green" : "red"}>{value === "up" ? "赞" : "踩"}</Tag> },
                  { title: "原因", dataIndex: "reason", render: (value) => value || "-" },
                  { title: "问题", dataIndex: "question", ellipsis: true },
                  { title: "回答", dataIndex: "answer", ellipsis: true },
                  { title: "用户", dataIndex: "user_id", render: (value) => value?.slice(0, 8) || "-" },
                  { title: "知识库", dataIndex: "knowledge_base_id", render: (value) => value?.slice(0, 8) || "-" },
                  { title: "时间", dataIndex: "created_at", render: formatTime },
                ]}
              />
            ),
          },
        ]}
      />
      <Modal
        title="创建质量待办"
        open={Boolean(issueDraft)}
        onOk={submitQualityIssue}
        confirmLoading={createQualityIssueMutation.isPending}
        onCancel={() => setIssueDraft(null)}
        width={760}
      >
        {issueDraft && (
          <Space direction="vertical" className="full-select">
            <Descriptions bordered column={1} size="small" items={[
              { key: "question", label: "问题", children: issueDraft.question },
              { key: "answer", label: "回答", children: <Typography.Paragraph ellipsis={{ rows: 4, expandable: true }}>{issueDraft.answer}</Typography.Paragraph> },
              { key: "feedback", label: "反馈", children: `${issueDraft.feedback_reason || "-"} ${issueDraft.feedback_comment || ""}` },
              { key: "sources", label: "引用", children: `${issueDraft.citation_count || 0}/${issueDraft.source_count || 0}` },
            ]} />
            <Form form={issueForm} layout="vertical">
              <Form.Item name="knowledge_base_id" hidden><Input /></Form.Item>
              <Form.Item name="assistant_message_id" hidden><Input /></Form.Item>
              <Form.Item name="issue_type" label="问题类型" rules={[{ required: true }]}>
                <Select options={qualityIssueTypeOptions()} />
              </Form.Item>
              <Form.Item name="priority" label="优先级" rules={[{ required: true }]}>
                <Select options={qualityIssuePriorityOptions()} />
              </Form.Item>
              <Form.Item name="assignee_user_id" label="负责人">
                <Select allowClear showSearch optionFilterProp="label" loading={qualityIssueAssignees.isLoading} options={assigneeOptions} />
              </Form.Item>
              <Form.Item name="resolution_note" label="备注">
                <Input.TextArea rows={3} maxLength={2000} />
              </Form.Item>
            </Form>
          </Space>
        )}
      </Modal>
      <Drawer rootClassName="wide-drawer" width="72vw" title="处理质量待办" open={Boolean(qualityIssue)} onClose={() => setQualityIssue(null)}>
        {qualityIssue && (
          <Space direction="vertical" className="full-select">
            <QualityIssueDetail row={qualityIssue} />
            <Form form={issueUpdateForm} layout="vertical">
              <Form.Item name="status" label="状态" rules={[{ required: true }]}>
                <Select options={qualityIssueStatusOptions()} />
              </Form.Item>
              <Form.Item name="priority" label="优先级" rules={[{ required: true }]}>
                <Select options={qualityIssuePriorityOptions()} />
              </Form.Item>
              <Form.Item name="assignee_user_id" label="负责人">
                <Select allowClear showSearch optionFilterProp="label" loading={qualityIssueAssignees.isLoading} options={assigneeOptions} />
              </Form.Item>
              <Form.Item name="resolution_note" label="处理说明">
                <Input.TextArea rows={4} maxLength={4000} />
              </Form.Item>
              <Button type="primary" onClick={submitQualityIssueUpdate} loading={updateQualityIssueMutation.isPending}>保存处理结果</Button>
            </Form>
          </Space>
        )}
      </Drawer>
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

function ChatOperationDetail({ row }) {
  return (
    <Space direction="vertical" size={16} className="full-select">
      <Descriptions
        bordered
        column={2}
        size="small"
        items={[
          { key: "request_id", label: "Request ID", children: row.request_id || "-" },
          { key: "kb", label: "知识库", children: row.knowledge_base_id },
          { key: "user", label: "用户", children: row.user_id },
          { key: "api_key", label: "API Key", children: row.api_key_id || "-" },
          { key: "model", label: "模型", children: row.model_name || "-" },
          { key: "latency", label: "耗时", children: row.latency_ms == null ? "-" : `${row.latency_ms}ms` },
          { key: "tokens", label: "估算 Token", children: row.total_tokens ?? "-" },
          { key: "feedback", label: "反馈", children: row.feedback_rating || "-" },
          { key: "retry", label: "自动重试", children: row.auto_retry_triggered ? `${row.retry_count} 次` : "否" },
          { key: "final_low", label: "最终低可信", children: row.final_low_confidence ? "是" : "否" },
        ]}
      />
      <Typography.Title level={5}>问题</Typography.Title>
      <Typography.Paragraph>{row.question}</Typography.Paragraph>
      <Typography.Title level={5}>回答</Typography.Title>
      <Typography.Paragraph className="answer-content">{row.answer}</Typography.Paragraph>
      {(row.retry_trace || []).length > 0 && (
        <>
          <Typography.Title level={5}>检索重试轨迹</Typography.Title>
          <Table
            rowKey={(item) => item.attempt_index}
            size="small"
            pagination={false}
            dataSource={row.retry_trace || []}
            columns={[
              { title: "#", dataIndex: "attempt_index", width: 60 },
              { title: "Top K", dataIndex: "top_k", width: 80 },
              { title: "重排数", dataIndex: "rerank_top_n", width: 90 },
              { title: "改写", dataIndex: "query_rewrite_enabled", width: 80, render: (value) => value ? "开" : "关" },
              { title: "重排", dataIndex: "rerank_enabled", width: 80, render: (value) => value ? "开" : "关" },
              { title: "状态", dataIndex: "answer_status", width: 130, render: answerStatusText },
              { title: "可信度", dataIndex: "confidence", width: 120, render: (value, item) => confidenceText(value, item.confidence_score) },
              { title: "引用", width: 80, render: (_, item) => `${item.citation_count || 0}/${item.source_count || 0}` },
              { title: "检索问题", dataIndex: "retrieval_query", ellipsis: true },
            ]}
          />
        </>
      )}
      <Typography.Title level={5}>引用来源</Typography.Title>
      <Table
        rowKey={(source, index) => `${source.chunk_id || source.source_index}-${index}`}
        size="small"
        pagination={false}
        dataSource={row.sources || []}
        columns={[
          { title: "引用", dataIndex: "source_index", width: 70, render: (value) => `[${value || "-"}]` },
          { title: "文件", dataIndex: "filename" },
          { title: "分块", dataIndex: "chunk_index", width: 80 },
          { title: "页码", dataIndex: "page_number", width: 80, render: (value) => value || "-" },
          { title: "分数", width: 120, render: (_, source) => scoreText(source.rerank_score ?? source.hybrid_score ?? source.vector_score) },
          { title: "片段", dataIndex: "snippet" },
        ]}
      />
      {row.feedback_comment && (
        <>
          <Typography.Title level={5}>反馈说明</Typography.Title>
          <Typography.Paragraph>{row.feedback_reason || "-"}：{row.feedback_comment}</Typography.Paragraph>
        </>
      )}
    </Space>
  );
}

function QualityIssueDetail({ row }) {
  return (
    <Space direction="vertical" size={16} className="full-select">
      <Descriptions
        bordered
        column={2}
        size="small"
        items={[
          { key: "id", label: "待办 ID", children: row.id },
          { key: "kb", label: "知识库", children: row.knowledge_base_id },
          { key: "type", label: "类型", children: qualityIssueTypeText(row.issue_type) },
          { key: "priority", label: "优先级", children: qualityIssuePriorityText(row.priority) },
          { key: "status", label: "状态", children: qualityIssueStatusText(row.status) },
          { key: "assignee", label: "负责人", children: row.assignee_user_id || "-" },
          { key: "feedback", label: "反馈", children: `${row.feedback_rating || "-"} / ${row.feedback_reason || "-"}` },
          { key: "resolved", label: "完成时间", children: formatTime(row.resolved_at) },
        ]}
      />
      <Typography.Title level={5}>问题</Typography.Title>
      <Typography.Paragraph>{row.question}</Typography.Paragraph>
      <Typography.Title level={5}>回答快照</Typography.Title>
      <Typography.Paragraph className="answer-content">{row.answer_snapshot}</Typography.Paragraph>
      {row.feedback_comment && (
        <>
          <Typography.Title level={5}>反馈说明</Typography.Title>
          <Typography.Paragraph>{row.feedback_comment}</Typography.Paragraph>
        </>
      )}
      {row.resolution_note && (
        <>
          <Typography.Title level={5}>处理说明</Typography.Title>
          <Typography.Paragraph>{row.resolution_note}</Typography.Paragraph>
        </>
      )}
      <Typography.Title level={5}>引用快照</Typography.Title>
      <Table
        rowKey={(source, index) => `${source.chunk_id || source.source_index}-${index}`}
        size="small"
        pagination={false}
        dataSource={row.sources_snapshot || []}
        columns={[
          { title: "引用", dataIndex: "source_index", width: 70, render: (value) => `[${value || "-"}]` },
          { title: "文件", dataIndex: "filename" },
          { title: "分块", dataIndex: "chunk_index", width: 80 },
          { title: "页码", dataIndex: "page_number", width: 80, render: (value) => value || "-" },
          { title: "分数", width: 120, render: (_, source) => scoreText(source.rerank_score ?? source.hybrid_score ?? source.vector_score) },
          { title: "片段", dataIndex: "snippet" },
        ]}
      />
    </Space>
  );
}

function EmptyText({ text }) {
  return <div className="empty-text">{text}</div>;
}

function WorkspaceLoading() {
  return <div className="workspace-loading">正在加载工作台...</div>;
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
  queryClient.invalidateQueries({ queryKey: ["document-tasks", kbId] });
  queryClient.invalidateQueries({ queryKey: ["queue-health", kbId] });
}

function hasActiveIngestWork(items) {
  return Array.isArray(items) && items.some((item) => ["pending", "running", "processing"].includes(item.status));
}

function activateOnKeyboard(event, action) {
  if (event.target !== event.currentTarget || !["Enter", " "].includes(event.key)) return;
  event.preventDefault();
  action();
}

function buildChatRecordsFromMessages(items) {
  const records = [];
  let pendingQuestion = null;
  for (const item of items) {
    if (item.role === "user") {
      pendingQuestion = item;
      continue;
    }
    if (item.role === "assistant") {
      records.unshift({
        conversation_id: item.conversation_id,
        user_message_id: pendingQuestion?.id || item.id,
        assistant_message_id: item.id,
        question: pendingQuestion?.content || "历史问题",
        answer: item.content,
        sources: item.metadata?.sources || [],
        confidence: item.metadata?.confidence || "medium",
        confidence_score: item.metadata?.confidence_score,
        answer_status: item.metadata?.answer_status || "unknown",
        citation_count: item.metadata?.citation_count || 0,
        citation_coverage: item.metadata?.citation_coverage || 0,
        retry_count: item.metadata?.retry_count || 0,
        retry_trace: item.metadata?.retry_trace || [],
        auto_retry_triggered: Boolean(item.metadata?.auto_retry_triggered),
        final_low_confidence: Boolean(item.metadata?.final_low_confidence),
      });
      pendingQuestion = null;
    }
  }
  if (pendingQuestion) {
    records.unshift({
      conversation_id: pendingQuestion.conversation_id,
      user_message_id: pendingQuestion.id,
      assistant_message_id: "",
      question: pendingQuestion.content,
      answer: "该问题尚未生成回答，可能是上次问答中断或失败。",
      sources: [],
      confidence: "low",
      confidence_score: null,
      answer_status: "interrupted",
      citation_count: 0,
      citation_coverage: 0,
      retry_count: 0,
      retry_trace: [],
      auto_retry_triggered: false,
      final_low_confidence: true,
    });
  }
  return records;
}

function renderAnswerWithCitations(answer = "", sources = [], onCitationClick) {
  const sourceByIndex = new Map((sources || []).map((source) => [Number(source.source_index), source]));
  const parts = String(answer).split(/([\[【][0-9,\s，、-]+[\]】])/g);
  return parts.map((part, index) => {
    const citations = extractCitationIndexes(part);
    if (!citations.length) return <React.Fragment key={`${part}-${index}`}>{part}</React.Fragment>;
    return (
      <React.Fragment key={`${part}-${index}`}>
        {citations.map((citation) => {
          const source = sourceByIndex.get(citation);
          return (
            <Button
              key={`${part}-${index}-${citation}`}
              className="citation-button"
              type="link"
              size="small"
              disabled={!source}
              onClick={() => source && onCitationClick(source)}
            >
              [{citation}]
            </Button>
          );
        })}
      </React.Fragment>
    );
  });
}

function extractCitationIndexes(text = "") {
  const match = String(text).match(/^[\[【]([0-9,\s，、-]+)[\]】]$/);
  if (!match) return [];
  const indexes = [];
  for (const part of match[1].trim().split(/[,，、\s]+/)) {
    if (!part) continue;
    if (part.includes("-")) {
      const [startText, endText] = part.split("-", 2);
      const start = Number(startText);
      const end = Number(endText);
      if (Number.isInteger(start) && Number.isInteger(end) && start > 0 && end >= start && end <= start + 10) {
        for (let value = start; value <= end; value += 1) indexes.push(value);
      }
    } else {
      const value = Number(part);
      if (Number.isInteger(value) && value > 0) indexes.push(value);
    }
  }
  return indexes;
}

function answerStatusText(value) {
  const labels = {
    supported: "引用已验证",
    no_sources: "无证据",
    citation_missing: "缺少引用",
    citation_invalid: "引用无效",
    citation_incomplete: "引用不完整",
    interrupted: "未完成",
    unknown: "未知/历史",
  };
  return labels[value] || value || "未知/历史";
}

function isFullAccessKb(item) {
  return Boolean(
    item?.has_full_access
    || (item?.can_manage_members && item?.can_manage_settings && item?.can_manage_api_keys)
  );
}

function isQualityIssueCandidate(row) {
  return Boolean(
    row?.assistant_message_id
    && (
      row.feedback_rating === "down"
      || row.final_low_confidence
      || row.confidence === "low"
      || (row.citation_count || 0) === 0
      || (row.source_count || 0) === 0
      || ["no_sources", "citation_missing", "citation_invalid", "citation_incomplete"].includes(row.answer_status)
    )
  );
}

function defaultIssueType(row) {
  if (row.feedback_reason === "permission_leak") return "permission_risk";
  if (row.feedback_reason === "content_outdated") return "outdated_content";
  if (row.feedback_reason === "source_mismatch") return "source_mismatch";
  if (row.feedback_reason === "source_missing" || row.answer_status === "citation_missing" || row.answer_status === "no_sources") return "missing_source";
  if (row.feedback_reason === "answer_wrong" || row.feedback_rating === "down") return "wrong_answer";
  return "other";
}

function qualityIssueTypeOptions() {
  return [
    { value: "wrong_answer", label: "答案错误" },
    { value: "missing_source", label: "缺少来源" },
    { value: "source_mismatch", label: "引用不匹配" },
    { value: "outdated_content", label: "内容过时" },
    { value: "permission_risk", label: "权限风险" },
    { value: "other", label: "其他" },
  ];
}

function qualityIssueTypeText(value) {
  return qualityIssueTypeOptions().find((item) => item.value === value)?.label || value || "-";
}

function qualityIssuePriorityOptions() {
  return [
    { value: "low", label: "低" },
    { value: "medium", label: "中" },
    { value: "high", label: "高" },
    { value: "urgent", label: "紧急" },
  ];
}

function qualityIssuePriorityText(value) {
  return qualityIssuePriorityOptions().find((item) => item.value === value)?.label || value || "-";
}

function qualityIssuePriorityColor(value) {
  return { low: "default", medium: "blue", high: "orange", urgent: "red" }[value] || "default";
}

function qualityIssueStatusOptions() {
  return [
    { value: "open", label: "待处理" },
    { value: "in_progress", label: "处理中" },
    { value: "resolved", label: "已解决" },
    { value: "ignored", label: "已忽略" },
  ];
}

function qualityIssueStatusText(value) {
  return qualityIssueStatusOptions().find((item) => item.value === value)?.label || value || "-";
}

function qualityIssueStatusColor(value) {
  return { open: "red", in_progress: "blue", resolved: "green", ignored: "default" }[value] || "default";
}

function readError(error) {
  const detail = error?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (detail?.message) return detail.message;
  return error?.message || "请求失败";
}

function compactParams(values = {}) {
  return Object.fromEntries(
    Object.entries(values).filter(([, value]) => value !== "" && value !== undefined && value !== null && value !== false)
  );
}

function formatTime(value) {
  if (!value) return "-";
  const d = new Date(value);
  if (isNaN(d.getTime())) return "-";
  return d.toLocaleString("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

function formatBytes(value) {
  if (value == null) return "-";
  const size = Number(value);
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function taskStatusText(value) {
  return {
    pending: "等待中",
    processing: "处理中",
    completed: "已完成",
    failed: "失败",
    cancelled: "已取消",
    deleted: "已删除",
  }[value] || value || "-";
}

function taskStatusColor(value) {
  return {
    pending: "gold",
    processing: "blue",
    completed: "green",
    failed: "red",
    cancelled: "default",
    deleted: "default",
  }[value] || "default";
}

function taskStageText(value) {
  return {
    waiting_worker: "等待 Worker",
    starting: "启动",
    parsing: "解析",
    chunking: "切分",
    embedding: "向量化",
    indexing: "索引中",
    persisting: "写入",
    ready: "可用",
    failed: "失败",
    cancelled: "已取消",
    deleted: "已删除",
  }[value] || value || "-";
}

function scoreText(value) {
  if (value == null) return "-";
  return Number(value).toFixed(3);
}

function splitUrls(value = "") {
  return String(value).split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
}

function planStatusColor(row) {
  if (row.severity === "pass") return "green";
  if (row.severity === "warning") return "orange";
  return "red";
}

function confidenceText(value, score) {
  const labels = { high: "高可信", medium: "中可信", low: "低可信" };
  const suffix = score == null ? "" : ` ${Number(score).toFixed(2)}`;
  return `${labels[value] || "中可信"}${suffix}`;
}

function confidenceColor(value) {
  if (value === "high") return "green";
  if (value === "low") return "orange";
  return "blue";
}

createRoot(document.getElementById("root")).render(<App />);
