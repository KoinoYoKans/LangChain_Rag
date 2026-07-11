import { useState } from "react";
import { DeleteOutlined, KeyOutlined } from "@ant-design/icons";
import {
  Button,
  DatePicker,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Modal,
  Select,
  Table,
  Tag,
  Typography,
  message,
} from "antd";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

export default function ApiKeyWorkspace({ api }) {
  const queryClient = useQueryClient();
  const [form] = Form.useForm();
  const [secret, setSecret] = useState("");
  const kbs = useQuery({ queryKey: ["kbs"], queryFn: async () => (await api.get("/knowledge-bases")).data.items || [] });
  const manageableKbs = (kbs.data || []).filter(isFullAccessKb);
  const keys = useQuery({ queryKey: ["apiKeys"], queryFn: async () => (await api.get("/api-keys")).data.items || [] });
  const createKey = useMutation({
    mutationFn: (values) => api.post("/api-keys", {
      ...values,
      expires_at: values.expires_at ? values.expires_at.toISOString() : null,
      daily_request_limit: values.daily_request_limit || null,
      daily_token_limit: values.daily_token_limit || null,
      purpose: values.purpose || null,
    }),
    onSuccess: ({ data }) => {
      setSecret(data.secret);
      form.resetFields();
      queryClient.invalidateQueries({ queryKey: ["apiKeys"] });
    },
    onError: (error) => message.error(readError(error)),
  });
  const deleteKey = useMutation({
    mutationFn: (id) => api.delete(`/api-keys/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["apiKeys"] });
      message.success("API Key 已禁用");
    },
    onError: (error) => message.error(readError(error)),
  });

  function confirmDeleteKey(row) {
    Modal.confirm({
      title: `禁用 API Key「${row.name}」？`,
      content: "禁用后使用该 Key 的三方项目将无法继续访问接口。",
      okText: "禁用",
      okButtonProps: { danger: true },
      cancelText: "取消",
      onOk: () => deleteKey.mutate(row.id),
    });
  }

  return (
    <section className="surface full">
      <PageTitle title="开放接口" subtitle="为其他项目预留 OpenAI 兼容问答接口和独立检索接口，Key 绑定具体用户与知识库。" />
      <Form form={form} layout="inline" onFinish={(values) => createKey.mutate(values)}>
        <Form.Item name="name" rules={[{ required: true }]}><Input prefix={<KeyOutlined />} placeholder="Key 名称" /></Form.Item>
        <Form.Item name="knowledge_base_id" rules={[{ required: true }]} className="kb-select">
          <Select placeholder="绑定知识库" options={manageableKbs.map((item) => ({ value: item.id, label: item.name }))} />
        </Form.Item>
        <Form.Item name="expires_at">
          <DatePicker showTime placeholder="过期时间" disabledDate={(current) => current && current.endOf("day").valueOf() < Date.now()} />
        </Form.Item>
        <Form.Item name="daily_request_limit"><InputNumber min={1} max={1000000} placeholder="每日请求" /></Form.Item>
        <Form.Item name="daily_token_limit"><InputNumber min={1} max={100000000} placeholder="每日Token" /></Form.Item>
        <Form.Item name="purpose"><Input placeholder="用途备注" /></Form.Item>
        <Button type="primary" htmlType="submit" disabled={!manageableKbs.length} loading={createKey.isPending}>创建 API Key</Button>
      </Form>
      <Descriptions className="api-doc" bordered column={1} items={[
        { key: "chat", label: "OpenAI 兼容", children: "POST /v1/chat/completions，Header: Authorization: Bearer rag-..." },
        { key: "retrieval", label: "检索接口", children: "POST /v1/knowledge/{knowledge_base_id}/retrieval" },
      ]} />
      <Table
        rowKey="id"
        dataSource={keys.data || []}
        scroll={{ x: "max-content" }}
        columns={[
          { title: "名称", dataIndex: "name" },
          { title: "前缀", dataIndex: "key_prefix" },
          { title: "状态", width: 100, render: (_, row) => <Tag color={apiKeyStatus(row).color}>{apiKeyStatus(row).text}</Tag> },
          { title: "知识库", dataIndex: "knowledge_base_id", render: (value) => kbs.data?.find((item) => item.id === value)?.name || value?.slice(0, 8) },
          { title: "今日请求", width: 110, render: (_, row) => quotaText(row.daily_request_count, row.daily_request_limit) },
          { title: "今日Token", width: 120, render: (_, row) => quotaText(row.daily_token_count, row.daily_token_limit) },
          { title: "过期时间", dataIndex: "expires_at", render: formatTime },
          { title: "用途", dataIndex: "purpose", ellipsis: true, render: (value) => value || "-" },
          { title: "最近使用", dataIndex: "last_used_at", render: formatTime },
          { title: "创建时间", dataIndex: "created_at", render: formatTime },
          { title: "操作", width: 90, render: (_, row) => <Button size="small" danger disabled={!row.is_active} icon={<DeleteOutlined />} onClick={() => confirmDeleteKey(row)} /> },
        ]}
      />
      <Modal title="API Key 只显示一次" open={Boolean(secret)} footer={<Button type="primary" onClick={() => setSecret("")}>我已保存</Button>} onCancel={() => setSecret("")}>
        <Input.TextArea value={secret} autoSize readOnly />
      </Modal>
    </section>
  );
}

function PageTitle({ title, subtitle }) {
  return <div className="page-title"><Typography.Title level={3}>{title}</Typography.Title><Typography.Text type="secondary">{subtitle}</Typography.Text></div>;
}

function isFullAccessKb(item) {
  return Boolean(item?.has_full_access || (item?.can_manage_members && item?.can_manage_settings && item?.can_manage_api_keys));
}

function readError(error) {
  const detail = error?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (detail?.message) return detail.message;
  return error?.message || "请求失败";
}

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

function quotaText(count = 0, limit) {
  return limit ? `${count || 0}/${limit}` : `${count || 0}/不限`;
}

function apiKeyStatus(row) {
  if (!row.is_active) return { text: "禁用", color: "red" };
  if (row.expires_at && new Date(row.expires_at).getTime() <= Date.now()) return { text: "已过期", color: "orange" };
  if (row.daily_request_limit && row.daily_request_count >= row.daily_request_limit) return { text: "请求超额", color: "red" };
  if (row.daily_token_limit && row.daily_token_count >= row.daily_token_limit) return { text: "Token超额", color: "red" };
  return { text: "可用", color: "green" };
}
