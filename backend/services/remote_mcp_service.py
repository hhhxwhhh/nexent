import logging

from fastmcp import Client

from consts.exceptions import MCPConnectionError, MCPNameIllegal
from database.remote_mcp_db import (
    create_mcp_record,
    delete_mcp_record_by_name_and_url,
    get_mcp_records_by_tenant,
    check_mcp_name_exists,
    update_mcp_status_by_name_and_url
)

logger = logging.getLogger("remote_mcp_service")


async def mcp_server_health(remote_mcp_server: str) -> bool:
    try:
        client = Client(remote_mcp_server)
        async with client:
            connected = client.is_connected()
            return connected
    except Exception as e:
        logger.error(f"Remote MCP server health check failed: {e}")
        raise MCPConnectionError("MCP connection failed")


async def add_remote_mcp_server_list(tenant_id: str,
                                     user_id: str,
                                     remote_mcp_server: str,
                                     remote_mcp_server_name: str):

    # check if MCP name already exists
    if check_mcp_name_exists(mcp_name=remote_mcp_server_name, tenant_id=tenant_id):
        logger.error(
            f"MCP name already exists, tenant_id: {tenant_id}, remote_mcp_server_name: {remote_mcp_server_name}")
        raise MCPNameIllegal("MCP name already exists")

    # check if the address is available
    if not await mcp_server_health(remote_mcp_server=remote_mcp_server):
        raise MCPConnectionError("MCP connection failed")

    # update the PG database record
    insert_mcp_data = {"mcp_name": remote_mcp_server_name,
                       "mcp_server": remote_mcp_server,
                       "status": True}
    create_mcp_record(
        mcp_data=insert_mcp_data, tenant_id=tenant_id, user_id=user_id)


async def delete_remote_mcp_server_list(tenant_id: str,
                                        user_id: str,
                                        remote_mcp_server: str,
                                        remote_mcp_server_name: str):
    # delete the record in the PG database
    delete_mcp_record_by_name_and_url(mcp_name=remote_mcp_server_name,
                                      mcp_server=remote_mcp_server,
                                      tenant_id=tenant_id,
                                      user_id=user_id)


async def get_remote_mcp_server_list(tenant_id: str):
    mcp_records = get_mcp_records_by_tenant(tenant_id=tenant_id)
    mcp_records_list = []

    for record in mcp_records:
        mcp_records_list.append({
            "remote_mcp_server_name": record["mcp_name"],
            "remote_mcp_server": record["mcp_server"],
            "status": record["status"]
        })
    return mcp_records_list


async def check_mcp_health_and_update_db(mcp_url, service_name, tenant_id, user_id):
    # check the health of the MCP server
    try:
        status = await mcp_server_health(remote_mcp_server=mcp_url)
    except Exception:
        status = False
    # update the status of the MCP server in the database
    update_mcp_status_by_name_and_url(
        mcp_name=service_name,
        mcp_server=mcp_url,
        tenant_id=tenant_id,
        user_id=user_id,
        status=status)
    if not status:
        raise MCPConnectionError("MCP connection failed")


async def get_tool_from_remote_mcp_server(mcp_server_name: str, remote_mcp_server: str) -> dict:
    """
    从远程MCP服务器获取工具信息
    
    Args:
        mcp_server_name: MCP服务器名称
        remote_mcp_server: 远程MCP服务器地址
        
    Returns:
        dict: 工具信息字典
    """
    try:
        # 创建MCP客户端
        client = Client(remote_mcp_server)
        async with client:
            # 获取工具列表
            tools = await client.list_tools()
            
            # 获取每个工具的详细信息
            tool_details = []
            for tool in tools:
                try:
                    # 获取工具详细信息
                    tool_detail = await client.get_tool(tool.name)
                    tool_details.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                        "output_schema": getattr(tool, 'output_schema', None)
                    })
                except Exception as e:
                    logger.warning(f"Failed to get details for tool {tool.name}: {str(e)}")
                    # 即使某个工具详情获取失败，也保留基本信息
                    tool_details.append({
                        "name": tool.name,
                        "description": tool.description,
                        "error": str(e)
                    })
            
            return {
                "status": "success",
                "mcp_server": mcp_server_name,
                "tools_count": len(tool_details),
                "tools": tool_details
            }
    except Exception as e:
        logger.error(f"Failed to get tools from remote MCP server {remote_mcp_server}: {str(e)}")
        raise MCPConnectionError(f"Failed to connect to MCP server: {str(e)}")
    

async def execute_tool_on_remote_mcp_server(
    mcp_server_name: str, 
    remote_mcp_server: str,
    tool_name: str,
    tool_params: dict
) -> dict:
    """
    在远程MCP服务器上执行工具
    
    Args:
        mcp_server_name: MCP服务器名称
        remote_mcp_server: 远程MCP服务器地址
        tool_name: 工具名称
        tool_params: 工具参数
        
    Returns:
        dict: 执行结果
    """
    try:
        # 创建MCP客户端
        client = Client(remote_mcp_server)
        async with client:
            # 执行工具
            result = await client.call_tool(tool_name, tool_params)
            
            return {
                "status": "success",
                "mcp_server": mcp_server_name,
                "tool_name": tool_name,
                "result": result
            }
    except Exception as e:
        logger.error(f"Failed to execute tool {tool_name} on remote MCP server {remote_mcp_server}: {str(e)}")
        return {
            "status": "error",
            "mcp_server": mcp_server_name,
            "tool_name": tool_name,
            "error": str(e)
        }
