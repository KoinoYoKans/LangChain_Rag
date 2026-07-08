import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ApiOutlined,
  AuditOutlined,
  BankOutlined,
  CloudUploadOutlined,
  DeleteOutlined,
  DislikeOutlined,
  EditOutlined,
  EyeOutlined,
  FileSearchOutlined,
  HistoryOutlined,
  KeyOutlined,
  LikeOutlined,
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
        {page === "knowledge" && <KnowledgeWorkspace api={api} user={user} />}
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

function KnowledgeWorkspace({ api, user }) {
  const queryClient = useQueryClient();
  const [selectedKb, setSelectedKb] = useState("");
  const [preview, setPreview] = useState(null);
  const [editingKb, setEditingKb] = useState(null);
  const [fileStatus, setFileStatus] = useState("");
  const [selectedFileIds, setSelectedFileIds] = useState([]);
  const [selectedJobIds, setSelectedJobIds] = useState([]);
  const [urlPlan, setUrlPlan] = useState(null);
  const [selectedUrlItemIds, setSelectedUrlItemIds] = useState([]);
  const [kbForm] = Form.useForm();
  const [editForm] = Form.useForm();
  const [memberForm] = Form.useForm();
  const [urlForm] = Form.useForm();
  const kbs = useQuery({ queryKey: ["kbs"], queryFn: async () => (await api.get("/knowledge-bases")).data.items || [] });
  const departments = useQuery({ queryKey: ["departments"], queryFn: async () => (await api.get("/departments")).data.items || [] });
  const users = useQuery({
    queryKey: ["users"],
    enabled: user?.role !== "member",
    queryFn: async () => (await api.get("/users")).data.items || [],
  });
  const activeKb = selectedKb || kbs.data?.[0]?.id || "";
  const activeKbRecord = (kbs.data || []).find((item) => item.id === activeKb);
  const members = useQuery({
    queryKey: ["kb-members", activeKb],
    enabled: Boolean(activeKb),
    queryFn: async () => (await api.get(`/knowledge-bases/${activeKb}/members`)).data.items || [],
  });
  const files = useQuery({
    queryKey: ["files", activeKb, fileStatus],
    enabled: Boolean(activeKb),
    queryFn: async () => (await api.get(`/knowledge-bases/${activeKb}/documents`, { params: fileStatus ? { status: fileStatus } : {} })).data.items || [],
  });
  const activeJobs = useQuery({
    queryKey: ["jobs", activeKb, "active"],
    enabled: Boolean(activeKb),
    refetchInterval: 2500,
    queryFn: async () => (await api.get(`/knowledge-bases/${activeKb}/ingest-jobs`, { params: { status: "active" } })).data.items || [],
  });
  const historyJobs = useQuery({
    queryKey: ["jobs", activeKb, "history"],
    enabled: Boolean(activeKb),
    refetchInterval: 5000,
    queryFn: async () => (await api.get(`/knowledge-bases/${activeKb}/ingest-jobs`, { params: { status: "history" } })).data.items || [],
  });
  const queueHealth = useQuery({
    queryKey: ["queue-health", activeKb],
    enabled: Boolean(activeKb),
    refetchInterval: 5000,
    queryFn: async () => (await api.get(`/knowledge-bases/${activeKb}/queue-health`)).data,
  });
  useEffect(() => {
    if (!selectedKb && kbs.data?.[0]?.id) setSelectedKb(kbs.data[0].id);
  }, [kbs.data, selectedKb]);
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
  const upsertMember = useMutation({
    mutationFn: (values) => api.put(`/knowledge-bases/${activeKb}/members`, values),
    onSuccess: () => {
      memberForm.resetFields();
      queryClient.invalidateQueries({ queryKey: ["kb-members", activeKb] });
      queryClient.invalidateQueries({ queryKey: ["kbs"] });
      message.success("成员权限已保存");
    },
    onError: (error) => message.error(readError(error)),
  });
  const removeMember = useMutation({
    mutationFn: (userId) => api.delete(`/knowledge-bases/${activeKb}/members/${userId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["kb-members", activeKb] });
      queryClient.invalidateQueries({ queryKey: ["kbs"] });
      message.success("成员已移除");
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
          <Button type="primary" htmlType="submit" loading={createKb.isPending}>创建知识库</Button>
        </Form>
        <div className="kb-list">
          {(kbs.data || []).map((item) => (
            <div className={activeKb === item.id ? "kb-item active" : "kb-item"} key={item.id} onClick={() => setSelectedKb(item.id)}>
              <Flex justify="space-between" align="start" gap={8}>
                <div>
                  <strong>{item.name}</strong>
                  <span>{item.visibility} · {item.completed_file_count}/{item.file_count} 文件 · {item.failed_job_count} 失败</span>
                </div>
                <Space onClick={(event) => event.stopPropagation()}>
                  <Button size="small" icon={<EditOutlined />} onClick={() => openEditKb(item)} />
                  <Button size="small" danger icon={<DeleteOutlined />} onClick={() => confirmDeleteKb(item)} />
                </Space>
              </Flex>
            </div>
          ))}
        </div>
      </div>
      <div className="surface main">
        <Flex justify="space-between" align="center" gap={12}>
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
        <Upload.Dragger {...uploadProps} disabled={!activeKb} className="upload-zone">
          <p><CloudUploadOutlined /></p>
          <p>拖拽或点击上传文档</p>
        </Upload.Dragger>
        <Form form={urlForm} className="url-form stacked-form" layout="vertical" onFinish={(values) => ingestUrl.mutate(values)}>
          <Form.Item name="urls" rules={[{ required: true }]} className="grow">
            <Input.TextArea rows={4} placeholder="每行一个 URL，先校验再确认入队" />
          </Form.Item>
          <Space wrap>
            <Form.Item name="skip_duplicates" initialValue={true} valuePropName="checked" noStyle>
              <Switch checkedChildren="跳过重复" unCheckedChildren="允许重复" />
            </Form.Item>
            <Button type="primary" htmlType="submit" loading={ingestUrl.isPending}>开始校验</Button>
            <Button disabled={!urlPlan || !selectedUrlItemIds.length} loading={commitUrlPlan.isPending} onClick={() => commitUrlPlan.mutate()}>确认入队</Button>
          </Space>
        </Form>
        {urlPlan && (
          <div className="import-plan">
            <Flex justify="space-between" align="center" className="table-tools">
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
        <Flex justify="space-between" align="center" className="table-tools">
          <Typography.Title level={5}>文件</Typography.Title>
          <Space wrap>
            <Button disabled={!selectedFileIds.length} icon={<ReloadOutlined />} onClick={() => batchReindexDocs.mutate(selectedFileIds)}>批量重建</Button>
            <Button disabled={!selectedFileIds.length} danger icon={<DeleteOutlined />} onClick={() => batchDeleteDocs.mutate(selectedFileIds)}>批量删除</Button>
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
          rowSelection={{ selectedRowKeys: selectedFileIds, onChange: setSelectedFileIds }}
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
                  <Button icon={<ReloadOutlined />} onClick={() => reindexDoc.mutate(row.id)}>重建</Button>
                  <Button danger icon={<DeleteOutlined />} onClick={() => deleteDoc.mutate(row.id)} />
                </Space>
              ),
            },
          ]}
        />
        <Typography.Title level={5}>入库任务</Typography.Title>
        <Space className="table-tools" wrap>
          <Button disabled={!selectedJobIds.length} onClick={() => batchRetryJobs.mutate(selectedJobIds)}>批量重试失败</Button>
          <Button disabled={!selectedJobIds.length} danger onClick={() => batchCancelJobs.mutate(selectedJobIds)}>批量取消</Button>
        </Space>
        <Tabs
          items={[
            {
              key: "active",
              label: `处理中 ${activeJobs.data?.length || 0}`,
              children: <JobTable jobs={activeJobs.data || []} loading={activeJobs.isLoading} retryJob={retryJob} cancelJob={cancelJob} selectedJobIds={selectedJobIds} setSelectedJobIds={setSelectedJobIds} />,
            },
            {
              key: "history",
              label: <span><HistoryOutlined /> 历史 {historyJobs.data?.length || 0}</span>,
              children: <JobTable jobs={historyJobs.data || []} loading={historyJobs.isLoading} retryJob={retryJob} cancelJob={cancelJob} selectedJobIds={selectedJobIds} setSelectedJobIds={setSelectedJobIds} />,
            },
          ]}
        />
        <Typography.Title level={5}>成员权限</Typography.Title>
        <Form form={memberForm} layout="inline" className="url-form" onFinish={(values) => upsertMember.mutate(values)}>
          <Form.Item name="user_id" rules={[{ required: true }]} className="kb-select">
            <Select
              showSearch
              placeholder="选择用户"
              optionFilterProp="label"
              options={(users.data || []).map((item) => ({ value: item.id, label: `${item.display_name} · ${item.email}` }))}
            />
          </Form.Item>
          <Form.Item name="role" rules={[{ required: true }]} initialValue="viewer">
            <Select style={{ width: 140 }} options={[
              { value: "viewer", label: "viewer" },
              { value: "editor", label: "editor" },
              { value: "owner", label: "owner" },
            ]} />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={upsertMember.isPending}>授权</Button>
        </Form>
        <Table
          rowKey="user_id"
          size="small"
          loading={members.isLoading}
          pagination={false}
          dataSource={members.data || []}
          columns={[
            { title: "成员", render: (_, row) => <Space direction="vertical" size={0}><strong>{row.display_name}</strong><span className="muted">{row.email}</span></Space> },
            { title: "角色", dataIndex: "role", width: 120, render: (value) => <Tag color={value === "owner" ? "purple" : value === "editor" ? "blue" : "default"}>{value}</Tag> },
            { title: "部门", dataIndex: "department_id", width: 120, render: (value) => value?.slice(0, 8) || "-" },
            { title: "加入时间", dataIndex: "created_at", width: 180, render: formatTime },
            {
              title: "操作",
              width: 90,
              render: (_, row) => row.role === "owner" ? "-" : <Button size="small" danger onClick={() => removeMember.mutate(row.user_id)}>移除</Button>,
            },
          ]}
        />
      </div>
      <DocumentPreview api={api} kbId={activeKb} file={preview} onClose={() => setPreview(null)} />
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
        </Form>
      </Modal>
    </section>
  );
}

function JobTable({ jobs, loading, retryJob, cancelJob, selectedJobIds, setSelectedJobIds }) {
  return (
    <Table
      rowKey="id"
      size="small"
      loading={loading}
      pagination={{ pageSize: 6 }}
      dataSource={jobs}
      rowSelection={{ selectedRowKeys: selectedJobIds, onChange: setSelectedJobIds }}
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
              {row.status === "failed" && <Button size="small" onClick={() => retryJob.mutate(row.id)}>重试</Button>}
              {["pending", "running"].includes(row.status) && <Button size="small" danger onClick={() => cancelJob.mutate(row.id)}>取消</Button>}
            </Space>
          ),
        },
      ]}
    />
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
  const queryClient = useQueryClient();
  const [kbId, setKbId] = useState("");
  const [conversationId, setConversationId] = useState("");
  const [messages, setMessages] = useState([]);
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
              { value: "unclear", label: "表达不清楚" },
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
            <div className={conversationId === item.id ? "conversation-item active" : "conversation-item"} key={item.id} onClick={() => openConversation(item)} role="button" tabIndex={0}>
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
              <Flex justify="space-between" gap={12} align="start">
                <Typography.Title level={5}>{item.question}</Typography.Title>
                <Space>
                  <Tag color={confidenceColor(item.confidence)}>{confidenceText(item.confidence, item.confidence_score)}</Tag>
                  {item.assistant_message_id && (
                    <>
                      <Button size="small" icon={<LikeOutlined />} onClick={() => submitFeedback(item, "up")} />
                      <Button size="small" icon={<DislikeOutlined />} onClick={() => dislikeAnswer(item)} />
                    </>
                  )}
                </Space>
              </Flex>
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
      </div>
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
        { key: "chat", label: "OpenAI 兼容", children: "POST /v1/chat/completions，Header: Authorization: Bearer rag-..." },
        { key: "retrieval", label: "检索接口", children: "POST /v1/knowledge/{knowledge_base_id}/retrieval" },
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
  const feedback = useQuery({ queryKey: ["feedback"], refetchInterval: 10000, queryFn: async () => (await api.get("/feedback")).data.items || [] });
  return (
    <section className="surface full">
      <PageTitle title="审计与反馈" subtitle="记录用户操作、请求上下文和回答质量反馈。" />
      <Tabs
        items={[
          {
            key: "audit",
            label: "审计日志",
            children: (
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
  queryClient.invalidateQueries({ queryKey: ["queue-health", kbId] });
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
    });
  }
  return records;
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
